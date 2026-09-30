"""Debug video: the paper outline and paper-frame axes drawn back onto every frame."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from .geometry import Intrinsics, Pose, paper_corners_world, project
from .pipeline import PAPER, TRACKED, FrameCorners, FramePose

SOURCE_BGR = {PAPER: (0, 200, 0), TRACKED: (0, 200, 255)}
AXIS_LENGTH_M = 0.1


def _draw(frame: np.ndarray, index: int, corners: FrameCorners, pose: FramePose, smoothed: Pose | None,
          intrinsics: Intrinsics, paper: np.ndarray) -> np.ndarray:
    image = frame.copy()
    if smoothed is not None:
        outline = project(intrinsics, smoothed, paper)
        cv2.polylines(image, [np.round(outline).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    if corners.corners is not None:
        colour = SOURCE_BGR.get(corners.source, (0, 0, 255))
        for k, corner in enumerate(corners.corners):
            cv2.circle(image, tuple(np.round(corner).astype(int)), 4, colour, 1, cv2.LINE_AA)
            cv2.putText(image, f"C{k}", tuple(np.round(corner).astype(int) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        colour, 1, cv2.LINE_AA)
    if pose.pose is not None:
        axes = project(intrinsics, pose.pose, np.array([[0, 0, 0], [AXIS_LENGTH_M, 0, 0], [0, AXIS_LENGTH_M, 0],
                                                        [0, 0, AXIS_LENGTH_M]], dtype=float))
        origin = tuple(np.round(axes[0]).astype(int))
        for end, colour in zip(axes[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.arrowedLine(image, origin, tuple(np.round(end).astype(int)), colour, 2, cv2.LINE_AA, tipLength=0.15)
    text = f"frame {index}  {pose.source}"
    if pose.pose is not None:
        height = pose.pose.translation[2]
        text += f"  reproj {pose.reprojection_px:.2f}px  camera {100 * height:.1f}cm above table"
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (4, 4), (12 + width, 12 + height), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 8 + height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def write_debug_video(path: Path, frames: Sequence[np.ndarray], fps: float, corners: Sequence[FrameCorners],
                      poses: Sequence[FramePose], smoothed: Sequence[Pose | None], intrinsics: Intrinsics,
                      paper: np.ndarray | None = None) -> None:
    """Green corners: paper measured. Orange: tracked. White outline: smoothed pose. Axes x red, y green, z blue."""
    paper = paper_corners_world() if paper is None else paper
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (intrinsics.width, intrinsics.height))
    for i, frame in enumerate(frames):
        writer.write(_draw(frame, i, corners[i], poses[i], smoothed[i], intrinsics, paper))
    writer.release()
