"""Frame-to-frame motion of the table plane in the image, from texture points on the table and paper."""

from __future__ import annotations

import cv2
import numpy as np

from .segmentation import FrameMasks

MIN_INLIERS = 15


def plane_homography(previous: FrameMasks, current: FrameMasks) -> np.ndarray | None:
    """Homography taking table-plane pixels of `previous` to `current`, or None if tracking fails.

    Only points on the table or the paper are tracked, so the hand, arm and tube do not pull it.
    """
    prev_gray = np.clip(previous.gray, 0, 255).astype(np.uint8)
    cur_gray = np.clip(current.gray, 0, 255).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=600, qualityLevel=0.005, minDistance=6,
                                     mask=previous.plane.astype(np.uint8))
    if points is None or len(points) < MIN_INLIERS:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    forward, status_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, points, None, **lk)
    backward, status_b, _ = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, forward, None, **lk)
    good = (status_f.ravel() == 1) & (status_b.ravel() == 1)
    good &= np.linalg.norm((points - backward).reshape(-1, 2), axis=1) < 0.5
    ends = forward.reshape(-1, 2)
    h, w = current.plane.shape
    inside = (ends[:, 0] >= 0) & (ends[:, 0] < w) & (ends[:, 1] >= 0) & (ends[:, 1] < h)
    good &= inside
    good[inside] &= current.plane[ends[inside, 1].astype(int), ends[inside, 0].astype(int)]
    if good.sum() < MIN_INLIERS:
        return None
    homography, inliers = cv2.findHomography(points[good], forward[good], cv2.RANSAC, 1.0)
    if homography is None or inliers.sum() < MIN_INLIERS:
        return None
    return homography
