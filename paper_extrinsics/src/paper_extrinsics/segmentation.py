"""Colour classification of a frame into reference paper and table pixels.

Works in OpenCV's 8-bit Lab, where a and b are centred at 128. The paper is bright and neutral,
the table is a yellowish beige; skin, the green tube, the dark sleeve and the orange skeleton
overlay are neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColourThresholds:
    paper_min_lightness: int = 150
    paper_max_a: int = 10  # |a - 128|
    paper_b: tuple[int, int] = (106, 132)  # neutral to bluish: auto white balance drifts during a clip
    table_lightness: tuple[int, int] = (60, 205)
    table_min_b: int = 133
    table_a: tuple[int, int] = (116, 134)
    # iPhone sharpening leaves a dark halo a few pixels wide between the paper and the table.
    halo_px: int = 5


@dataclass
class FrameMasks:
    gray: np.ndarray  # float32 lightness, for edge refinement and tracking
    paper: np.ndarray  # bool
    table: np.ndarray  # bool
    halo_px: int = 5

    @property
    def plane(self) -> np.ndarray:
        """Pixels known to lie on the table plane: table or paper, away from anything else."""
        plane = (self.paper | self.table).astype(np.uint8)
        return cv2.erode(plane, np.ones((7, 7), np.uint8)).astype(bool)


def classify(frame_bgr: np.ndarray, th: ColourThresholds = ColourThresholds()) -> FrameMasks:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = (lab[..., i].astype(np.int16) for i in range(3))
    paper = (
        (lightness >= th.paper_min_lightness)
        & (np.abs(a - 128) <= th.paper_max_a)
        & (b >= th.paper_b[0])
        & (b <= th.paper_b[1])
    )
    table = (
        (lightness >= th.table_lightness[0])
        & (lightness <= th.table_lightness[1])
        & (b >= th.table_min_b)
        & (a >= th.table_a[0])
        & (a <= th.table_a[1])
    )
    kernel = np.ones((3, 3), np.uint8)
    paper = cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel).astype(bool)
    table = cv2.morphologyEx(table.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool) & ~paper
    return FrameMasks(gray=lab[..., 0].astype(np.float32), paper=paper, table=table, halo_px=th.halo_px)


def paper_table_boundary(masks: FrameMasks) -> np.ndarray:
    """(N, 2) x, y of paper pixels that touch the table: the only boundary pixels trusted as paper edges."""
    kernel = np.ones((3, 3), np.uint8)
    boundary = masks.paper & ~cv2.erode(masks.paper.astype(np.uint8), kernel).astype(bool)
    near_table = cv2.dilate(masks.table.astype(np.uint8), kernel, iterations=masks.halo_px).astype(bool)
    ys, xs = np.nonzero(boundary & near_table)
    return np.column_stack([xs, ys]).astype(np.float64)
