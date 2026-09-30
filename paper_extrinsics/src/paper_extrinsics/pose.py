"""Camera pose from the four image corners of the reference paper."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import Intrinsics, Pose, paper_corners_world, project, rotation_angle_deg

# When the two planar solutions fit about equally well, the one closer to the previous frame wins.
AMBIGUOUS_ERROR_RATIO = 1.5


def reprojection_rms(intrinsics: Intrinsics, pose: Pose, corners: np.ndarray, paper: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((project(intrinsics, pose, paper) - corners) ** 2, axis=1))))


def solve_pose(corners: np.ndarray, intrinsics: Intrinsics, previous: Pose | None = None,
               paper: np.ndarray | None = None) -> tuple[Pose, float] | None:
    """Pose and reprojection RMS in pixels, or None if no solution puts the camera above the table."""
    paper = paper_corners_world() if paper is None else paper
    image = np.ascontiguousarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    k = intrinsics.matrix()
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(paper, image, k, None, flags=cv2.SOLVEPNP_IPPE)
    solutions = []
    for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
        rvec, tvec = cv2.solvePnPRefineLM(paper, image, k, None, rvec.copy(), tvec.copy())
        pose = Pose.from_opencv(rvec, tvec)
        if pose.translation[2] <= 0:
            continue
        solutions.append((pose, reprojection_rms(intrinsics, pose, corners, paper)))
    if not solutions:
        return None
    solutions.sort(key=lambda s: s[1])
    if previous is not None and len(solutions) > 1 and solutions[1][1] < AMBIGUOUS_ERROR_RATIO * solutions[0][1] + 0.1:
        return min(solutions[:2], key=lambda s: rotation_angle_deg(s[0].rotation, previous.rotation))
    return solutions[0]
