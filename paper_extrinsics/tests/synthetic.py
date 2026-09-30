"""Renders a synthetic human demonstration: a handheld camera looking down at an A4 sheet on a textured table."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from paper_extrinsics.geometry import PAPER_LENGTH_M, PAPER_WIDTH_M, Intrinsics, Pose

# BGR colours sampled from the real clip.
TABLE_BGR = np.array([110, 145, 156], dtype=np.float32)
PAPER_BGR = np.array([214, 211, 207], dtype=np.float32)
TUBE_BGR = (8, 147, 59)
SKIN_BGR = (91, 117, 154)
OVERLAY_BGR = (40, 110, 250)

TEXTURE_MM_PER_PX = 1.0
TEXTURE_EXTENT_M = (1.4, 1.2)  # along world x, along world y

# Tube over corner C3 (+x, -y): both adjacent edges stay partly visible.
CORNER_OCCLUSION_FRAMES = range(40, 55)
CORNER_TUBE_M = np.array([[0.10, -0.15], [0.20, -0.15], [0.20, -0.07], [0.10, -0.07]])
# Tube along the whole +x short edge: that edge cannot be measured.
EDGE_OCCLUSION_FRAMES = range(60, 75)
EDGE_TUBE_M = np.array([[0.125, -0.16], [0.175, -0.16], [0.175, 0.16], [0.125, 0.16]])
# A hand resting on the table next to the paper, in every frame.
HAND_M = np.array([[-0.05, -0.30], [0.08, -0.30], [0.10, -0.18], [-0.04, -0.17]])


@dataclass
class SyntheticClip:
    frames: list[np.ndarray]
    intrinsics: Intrinsics
    poses: list[Pose]
    fps: float = 24.0
    notes: dict = field(default_factory=dict)


def _table_texture(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns the table+paper texture and the 3x3 map from texture pixel (u, v, 1) to world (x, y, 1)."""
    s = TEXTURE_MM_PER_PX / 1000.0
    rows = int(TEXTURE_EXTENT_M[1] / s)
    cols = int(TEXTURE_EXTENT_M[0] / s)
    x0, y0 = -TEXTURE_EXTENT_M[0] / 2, -TEXTURE_EXTENT_M[1] / 2
    # Column u runs along world x, row v along world y.
    texture_to_world = np.array([[s, 0, x0], [0, s, y0], [0, 0, 1]])

    shade = gaussian_filter(rng.normal(size=(rows, cols)), 60)
    shade = 12 * shade / np.abs(shade).max()
    table = TABLE_BGR[None, None, :] + shade[..., None]
    grain = gaussian_filter(rng.normal(size=(rows, cols)), 1.2)
    table += 25 * (grain / grain.std())[..., None]
    for _ in range(4000):
        u, v = rng.integers(0, cols), rng.integers(0, rows)
        cv2.circle(table, (int(u), int(v)), int(rng.integers(2, 6)), (TABLE_BGR * rng.uniform(0.6, 0.85)).tolist(), -1)

    v, u = np.mgrid[0:rows, 0:cols]
    x, y = x0 + u * s, y0 + v * s
    # Fraction of each texel covered by the sheet, so its edges sit exactly where they should.
    cover_x = np.clip((PAPER_LENGTH_M / 2 - np.abs(x)) / s + 0.5, 0, 1)
    cover_y = np.clip((PAPER_WIDTH_M / 2 - np.abs(y)) / s + 0.5, 0, 1)
    on_paper = (cover_x * cover_y)[..., None]
    paper_shade = -18 * np.clip((y + 0.02) / 0.1, 0, 1)  # a soft shadow over one side of the sheet
    paper = PAPER_BGR[None, None, :] + paper_shade[..., None]
    texture = on_paper * paper + (1 - on_paper) * table
    return np.clip(texture, 0, 255).astype(np.uint8), texture_to_world


def _plane_to_image(intrinsics: Intrinsics, pose: Pose) -> np.ndarray:
    rvec, tvec = pose.to_opencv()
    r_cw, _ = cv2.Rodrigues(rvec)
    return intrinsics.matrix() @ np.column_stack([r_cw[:, 0], r_cw[:, 1], tvec.ravel()])


def _fill_world_polygon(image, intrinsics, pose, polygon_m, colour):
    h = _plane_to_image(intrinsics, pose)
    pts = cv2.perspectiveTransform(polygon_m.reshape(-1, 1, 2).astype(np.float64), h)
    cv2.fillPoly(image, [np.round(pts * 16).astype(np.int32)], colour, lineType=cv2.LINE_AA, shift=4)


def camera_trajectory(n: int) -> list[Pose]:
    """A handheld camera about 0.62 m above the sheet, tilting up to about 20 degrees.

    In the first frame the top of the image faces world +x, so the paper frame convention
    (x toward the top of the first frame) matches this ground truth.
    """
    # Camera axes in world: image right = world -y, image down = world -x, optical axis = world -z.
    looking_down = np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).astype(float)
    poses = []
    for i in range(n):
        phase = 2 * np.pi * i / n
        tilt = Rotation.from_euler(
            "zxy",
            [10 * np.sin(phase), 20 * np.sin(phase / 0.7), 15 * np.cos(phase / 0.9) - 15],
            degrees=True,
        ).as_matrix()
        r_wc = looking_down @ tilt
        target = np.array([0.02 * np.sin(phase), 0.015 * np.cos(phase), 0.0])
        distance = 0.62 + 0.05 * np.sin(phase * 1.3)
        poses.append(Pose(r_wc, target - distance * r_wc[:, 2]))
    return poses


def make_clip(n_frames: int = 90, focal_px: float = 660.0, seed: int = 0) -> SyntheticClip:
    rng = np.random.default_rng(seed)
    intrinsics = Intrinsics.centred(focal_px, 832, 480)
    texture, texture_to_world = _table_texture(rng)
    poses = camera_trajectory(n_frames)
    frames = []
    for i, pose in enumerate(poses):
        h = _plane_to_image(intrinsics, pose) @ texture_to_world
        image = cv2.warpPerspective(texture, h, (intrinsics.width, intrinsics.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        _fill_world_polygon(image, intrinsics, pose, HAND_M, SKIN_BGR)
        if i in CORNER_OCCLUSION_FRAMES:
            _fill_world_polygon(image, intrinsics, pose, CORNER_TUBE_M, TUBE_BGR)
        if i in EDGE_OCCLUSION_FRAMES:
            _fill_world_polygon(image, intrinsics, pose, EDGE_TUBE_M, TUBE_BGR)
        if i % 3 == 0:  # skeleton overlay line crossing the sheet
            cv2.line(image, (300, 100), (520, 380), OVERLAY_BGR, 2, cv2.LINE_AA)
        image = cv2.GaussianBlur(image, (0, 0), 0.7)
        # Phone-style sharpening (dark halo outside bright edges) and a white balance drift.
        image = cv2.addWeighted(image, 1.8, cv2.GaussianBlur(image, (0, 0), 2.0), -0.8, 0)
        drift = 8 * np.sin(2 * np.pi * i / n_frames)
        image = image.astype(np.float32) + np.array([drift, 0, -drift], dtype=np.float32)
        image = np.clip(image + rng.normal(0, 1.5, image.shape), 0, 255).astype(np.uint8)
        frames.append(image)
    return SyntheticClip(frames=frames, intrinsics=intrinsics, poses=poses)
