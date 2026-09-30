"""Finding the reference paper's outline in one frame, as four fitted edge lines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import corners_from_edges, line_through
from .segmentation import FrameMasks, paper_table_boundary

MIN_COVERAGE = 0.3  # fraction of the edge's length that must show against the table
INITIAL_MIN_COVERAGE = 0.7  # the frame that fixes the paper frame must show the whole sheet
COVERAGE_BINS = 20
INLIER_PX = 1.5


@dataclass
class EdgeFit:
    line: np.ndarray | None  # (a, b, c), None when the edge was not measured
    inliers: int
    coverage: float

    @property
    def measured(self) -> bool:
        return self.line is not None


def _total_least_squares(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre, full_matrices=False)
    normal = vt[1]
    return np.array([normal[0], normal[1], -normal @ centre])


def _ransac_line(points: np.ndarray, rng: np.random.Generator, iterations: int = 150) -> np.ndarray:
    best, best_count = None, -1
    for _ in range(iterations):
        i, j = rng.choice(len(points), 2, replace=False)
        if np.allclose(points[i], points[j]):
            continue
        line = line_through(points[i], points[j])
        count = int(np.sum(np.abs(points @ line[:2] + line[2]) < INLIER_PX))
        if count > best_count:
            best, best_count = line, count
    inliers = np.abs(points @ best[:2] + best[2]) < INLIER_PX
    return inliers


def _refine_subpixel(gray: np.ndarray, line: np.ndarray, points: np.ndarray, half_width: float = 3.0) -> np.ndarray:
    """Moves each edge point along the line's normal to the strongest lightness step, to a fraction of a pixel."""
    normal = line[:2]
    feet = points - np.outer(points @ normal + line[2], normal)
    if len(feet) > 300:
        feet = feet[np.linspace(0, len(feet) - 1, 300).astype(int)]
    offsets = np.arange(-half_width, half_width + 1e-9, 0.25)
    samples = feet[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    profile = map_coordinates(gray, [samples[..., 1].ravel(), samples[..., 0].ravel()], order=1, mode="nearest")
    gradient = np.abs(np.diff(profile.reshape(len(feet), -1), axis=1))
    peak = np.argmax(gradient, axis=1)
    inner = (peak > 0) & (peak < gradient.shape[1] - 1)
    rows = np.arange(len(feet))[inner]
    g0, g1, g2 = gradient[rows, peak[inner] - 1], gradient[rows, peak[inner]], gradient[rows, peak[inner] + 1]
    denom = g0 - 2 * g1 + g2
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (g0 - g2) / denom, 0.0)
    offset = offsets[0] + 0.25 * (peak[inner] + 0.5 + shift)
    strong = g1 > 0.5 * np.median(g1)
    return feet[inner][strong] + offset[strong, None] * normal


def fit_edge(masks: FrameMasks, candidates: np.ndarray, p: np.ndarray, q: np.ndarray, band_px: float,
             min_coverage: float, rng: np.random.Generator) -> EdgeFit:
    """Fits the paper edge expected between p and q from paper/table boundary pixels near it."""
    length = np.linalg.norm(q - p)
    direction = (q - p) / length
    normal = np.array([-direction[1], direction[0]])
    rel = candidates - p
    along = rel @ direction / length
    across = rel @ normal
    margin = (band_px + 2) / length  # keep clear of the neighbouring edges near the corners
    near = (np.abs(across) < band_px) & (along > margin) & (along < 1 - margin)
    points = candidates[near]
    if len(points) < max(20, 0.15 * length):
        return EdgeFit(None, len(points), 0.0)

    inliers = _ransac_line(points, rng)
    points = points[inliers]
    coverage = len(np.unique(np.clip(((points - p) @ direction / length * COVERAGE_BINS).astype(int), 0, COVERAGE_BINS - 1)))
    coverage /= COVERAGE_BINS
    if coverage < min_coverage or len(points) < max(20, 0.15 * length):
        return EdgeFit(None, len(points), coverage)

    line = _total_least_squares(points)
    refined = _refine_subpixel(masks.gray, line, points)
    if len(refined) >= 10:
        line = _total_least_squares(refined)
        residual = np.abs(refined @ line[:2] + line[2])
        keep = residual < max(0.5, 3 * np.median(residual))
        line = _total_least_squares(refined[keep])
    return EdgeFit(line, len(points), coverage)


def fit_edges(masks: FrameMasks, predicted_corners: np.ndarray, band_px: float = 12.0,
              min_coverage: float = MIN_COVERAGE, seed: int = 0) -> list[EdgeFit]:
    rng = np.random.default_rng(seed)
    candidates = paper_table_boundary(masks)
    return [fit_edge(masks, candidates, predicted_corners[i], predicted_corners[(i + 1) % 4], band_px, min_coverage, rng)
            for i in range(4)]


def canonical_order(corners: np.ndarray) -> np.ndarray:
    """Orders four image corners as C0..C3 of the paper frame (see geometry.paper_corners_world).

    Seen from above the table, C0..C3 is counter-clockwise, which in image coordinates (y down) is a
    negative shoelace area. Edge 3 (C3 -> C0) is the short edge at +x, and +x is taken toward the top
    of the image: of the two short edges, the one higher in the image.
    """
    corners = np.asarray(corners, dtype=np.float64)
    x, y = corners[:, 0], corners[:, 1]
    area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area > 0:
        corners = corners[::-1]
    lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    # Edge k joins corners k and k+1; the short edges are {0, 2} or {1, 3}.
    short = [0, 2] if lengths[0] + lengths[2] < lengths[1] + lengths[3] else [1, 3]
    mid_y = {k: (corners[k, 1] + corners[(k + 1) % 4, 1]) / 2 for k in short}
    top = min(short, key=mid_y.get)
    # Rotate so that the top short edge becomes edge 3, i.e. starts at index 3.
    return np.roll(corners, -((top + 1) % 4), axis=0)


def find_initial_corners(masks: FrameMasks) -> np.ndarray | None:
    """Corners of a fully visible paper, in canonical order, or None."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(masks.paper.astype(np.uint8), connectivity=8)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.01 * masks.paper.size:
        return None
    contours, _ = cv2.findContours((labels == biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(quad) != 4:
        return None
    corners = canonical_order(quad.astype(np.float64))
    fits = fit_edges(masks, corners, min_coverage=INITIAL_MIN_COVERAGE)
    if not all(f.measured for f in fits):
        return None
    return corners_from_edges([f.line for f in fits])
