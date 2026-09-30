"""One focal length for the camera, from the reference paper's shape in many frames (ADR 0001)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .geometry import Intrinsics, paper_corners_world
from .pose import solve_pose

# iPhone 1x main camera in standard (stabilised) video, scaled to 832 px wide.
PRIOR_FOCAL_PX_AT_832 = 700.0
PRIOR_SIGMA_PX_AT_832 = 100.0
BOUNDS_PX_AT_832 = (580.0, 890.0)


@dataclass(frozen=True)
class FocalEstimate:
    focal_px: float
    sigma_px: float
    frames: int
    corner_noise_px: float
    at_bound: bool = False  # the estimate sits on the edge of the allowed range


def _squared_residuals(focal_px, corner_sets, size, pixel_aspect, paper) -> float:
    intrinsics = Intrinsics.centred(focal_px, size[0], size[1], pixel_aspect)
    total = 0.0
    for corners in corner_sets:
        solved = solve_pose(corners, intrinsics, paper=paper)
        total += 4 * solved[1] ** 2 if solved is not None else 1e6
    return total


def estimate_focal(corner_sets: Sequence[np.ndarray], size: tuple[int, int], pixel_aspect: float = 1.0,
                   paper: np.ndarray | None = None) -> FocalEstimate:
    """Maximum a posteriori focal length (px) over all frames, with its standard deviation.

    Every frame gets its own pose; the focal length is shared. The iPhone prior keeps the estimate
    sensible when the views are nearly top-down and the data barely constrain it.
    """
    paper = paper_corners_world() if paper is None else paper
    scale = size[0] / 832.0
    prior, prior_sigma = PRIOR_FOCAL_PX_AT_832 * scale, PRIOR_SIGMA_PX_AT_832 * scale
    low, high = BOUNDS_PX_AT_832[0] * scale, BOUNDS_PX_AT_832[1] * scale
    if not corner_sets:
        return FocalEstimate(prior, prior_sigma, 0, float("nan"), False)

    def rss(f):
        return _squared_residuals(f, corner_sets, size, pixel_aspect, paper)

    grid = np.linspace(low, high, 63)
    values = np.array([rss(f) for f in grid])
    best = grid[np.argmin(values)]
    step = grid[1] - grid[0]
    fit = minimize_scalar(rss, bounds=(max(low, best - step), min(high, best + step)), method="bounded",
                          options={"xatol": 0.05})
    # Each frame has 8 corner coordinates and 6 pose parameters; one focal length is shared.
    dof = max(1, 2 * len(corner_sets) - 1)
    noise_var = max(fit.fun / dof, 1e-6)

    def cost(f):
        return rss(f) / noise_var + ((f - prior) / prior_sigma) ** 2

    best = grid[np.argmin(values / noise_var + ((grid - prior) / prior_sigma) ** 2)]
    posterior = minimize_scalar(cost, bounds=(max(low, best - step), min(high, best + step)),
                                method="bounded", options={"xatol": 0.05})
    f, h = posterior.x, 1.0
    curvature = (cost(f + h) - 2 * cost(f) + cost(f - h)) / h**2
    sigma = float(np.sqrt(2 / curvature)) if curvature > 0 else prior_sigma
    at_bound = min(f - low, high - f) < 0.5
    return FocalEstimate(float(f), sigma, len(corner_sets), float(np.sqrt(noise_var)), at_bound)
