"""Paper frame, intrinsics, camera poses and 2D line helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

PAPER_LENGTH_M = 0.297
PAPER_WIDTH_M = 0.210


def paper_corners_world(length: float = PAPER_LENGTH_M, width: float = PAPER_WIDTH_M) -> np.ndarray:
    """Corners C0..C3 of the reference paper in the paper frame (z = 0).

    C0 = (+x, +y), C1 = (-x, +y), C2 = (-x, -y), C3 = (+x, -y): counter-clockwise seen from above.
    Edge i runs from Ci to C(i+1); edge 3 (C3 -> C0) is the short edge at +x.
    """
    hx, hy = length / 2, width / 2
    return np.array([[hx, hy, 0], [-hx, hy, 0], [-hx, -hy, 0], [hx, -hy, 0]], dtype=np.float64)


def stretched_16_9_aspect(width: int, height: int) -> float:
    """fy / fx of a 16:9 video resized, without cropping, to width x height (1.0256 for 832x480)."""
    return (16 / 9) * height / width


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def centred(cls, focal_px: float, width: int, height: int, pixel_aspect: float = 1.0) -> Intrinsics:
        """Principal point at the image centre; fy = pixel_aspect * fx."""
        return cls(focal_px, focal_px * pixel_aspect, (width - 1) / 2, (height - 1) / 2, width, height)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


@dataclass(frozen=True)
class Pose:
    """Camera pose T_world_cam: maps camera-frame points into the paper frame (p_world = R p_cam + t)."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def from_opencv(cls, rvec: np.ndarray, tvec: np.ndarray) -> Pose:
        r_cw, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        r_wc = r_cw.T
        return cls(r_wc, -r_wc @ np.asarray(tvec, dtype=np.float64).ravel())

    @classmethod
    def from_quaternion(cls, xyzw: np.ndarray, translation: np.ndarray) -> Pose:
        return cls(Rotation.from_quat(xyzw).as_matrix(), np.asarray(translation, dtype=np.float64))

    def to_opencv(self) -> tuple[np.ndarray, np.ndarray]:
        """rvec, tvec of the world-to-camera transform, as OpenCV's projection functions expect."""
        r_cw = self.rotation.T
        rvec, _ = cv2.Rodrigues(r_cw)
        return rvec, (-r_cw @ self.translation).reshape(3, 1)

    def quaternion_xyzw(self) -> np.ndarray:
        return Rotation.from_matrix(self.rotation).as_quat()

    def apply(self, points_cam: np.ndarray) -> np.ndarray:
        return points_cam @ self.rotation.T + self.translation


def project(intrinsics: Intrinsics, pose: Pose, points_world: np.ndarray) -> np.ndarray:
    rvec, tvec = pose.to_opencv()
    pixels, _ = cv2.projectPoints(np.asarray(points_world, dtype=np.float64), rvec, tvec, intrinsics.matrix(), None)
    return pixels.reshape(-1, 2)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# 2D lines are (a, b, c) with a*x + b*y + c = 0 and (a, b) a unit normal.


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    line = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    return line / np.hypot(line[0], line[1])


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    x = np.cross(l1, l2)
    return x[:2] / x[2]


def corners_from_edges(lines: list[np.ndarray]) -> np.ndarray:
    """Corner Ci lies on edge i-1 and edge i."""
    return np.array([intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)])


def edges_from_corners(corners: np.ndarray) -> list[np.ndarray]:
    return [line_through(corners[i], corners[(i + 1) % 4]) for i in range(4)]
