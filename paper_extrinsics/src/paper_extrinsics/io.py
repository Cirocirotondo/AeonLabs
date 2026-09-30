"""Reading videos and hand pose estimator CSVs; writing camera poses and intrinsics."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .focal import FocalEstimate
from .geometry import Intrinsics, Pose
from .pipeline import FramePose


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


@dataclass(frozen=True)
class SavedIntrinsics:
    intrinsics: Intrinsics
    focal_sigma_px: float
    source: str


def save_intrinsics(path: Path, intrinsics: Intrinsics, estimate: FocalEstimate | None, source: str,
                    clips: Sequence[str]) -> None:
    data = {
        "fx": intrinsics.fx, "fy": intrinsics.fy, "cx": intrinsics.cx, "cy": intrinsics.cy,
        "width": intrinsics.width, "height": intrinsics.height, "distortion": None,
        "focal_sigma_px": None if estimate is None else estimate.sigma_px,
        "source": source,
        "frames_used": None if estimate is None else estimate.frames,
        "corner_noise_px": None if estimate is None else estimate.corner_noise_px,
        "clips": list(clips),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_intrinsics(path: Path) -> SavedIntrinsics:
    data = json.loads(path.read_text())
    intrinsics = Intrinsics(data["fx"], data["fy"], data["cx"], data["cy"], data["width"], data["height"])
    return SavedIntrinsics(intrinsics, data.get("focal_sigma_px") or float("nan"), f"loaded from {path}")


def _pose_columns(pose: Pose | None) -> list[str]:
    if pose is None:
        return ["nan"] * 7
    return [f"{v:.6f}" for v in (*pose.translation, *pose.quaternion_xyzw())]


def write_poses_csv(path: Path, clip_name: str, fps: float, poses: Sequence[FramePose],
                    smoothed: Sequence[Pose | None], intrinsics: Intrinsics, focal_line: str,
                    smoothing_line: str, extra_header: Sequence[str] = ()) -> None:
    header = [
        f"{clip_name}: camera pose per frame from the A4 reference paper, {len(poses)} frames at {fps:g} fps",
        "pose: T_world_cam, maps camera-frame points into the paper frame (p_world = R p_cam + t); metres; quaternion xyzw",
        "paper frame: origin at the centre of the A4 sheet, z up out of the table, x along the long edge toward the top of the first frame",
        "camera frame: x right, y down, z away from the camera (OpenCV), as in the hand pose estimator's CSV",
        f"intrinsics: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f} px, "
        f"{intrinsics.width}x{intrinsics.height}, no distortion; {focal_line}",
        "source: paper = all four edges measured; tracked = carried over by tracking the table; failed = no pose",
        smoothing_line,
        *extra_header,
    ]
    names = ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"]
    columns = ["frame", "time_s", "source", "reprojection_px", *names, *(f"{c}_smooth" for c in names)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        for line in header:
            f.write(f"# {line}\n")
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, (p, s) in enumerate(zip(poses, smoothed)):
            writer.writerow([i, f"{i / fps:.6f}", p.source, f"{p.reprojection_px:.3f}", *_pose_columns(p.pose), *_pose_columns(s)])


def read_poses_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@dataclass(frozen=True)
class PosesHeader:
    intrinsics: Intrinsics
    paper_size_m: tuple[float, float]
    fps: float


def read_poses_header(path: Path) -> PosesHeader:
    """Intrinsics, paper size and frame rate recorded in a camera poses CSV's header."""
    text = "".join(line for line in path.open() if line.startswith("#"))
    k = re.search(r"fx=([\d.]+) fy=([\d.]+) cx=([\d.]+) cy=([\d.]+) px, (\d+)x(\d+)", text)
    if k is None:
        raise ValueError(f"no intrinsics in the header of {path}")
    fx, fy, cx, cy = (float(v) for v in k.groups()[:4])
    intrinsics = Intrinsics(fx, fy, cx, cy, int(k.group(5)), int(k.group(6)))
    paper = re.search(r"reference paper: ([\d.]+) x ([\d.]+) mm", text)
    size = (float(paper.group(1)) / 1000, float(paper.group(2)) / 1000) if paper else (0.297, 0.210)
    fps = re.search(r"frames at ([\d.]+) fps", text)
    return PosesHeader(intrinsics, size, float(fps.group(1)) if fps else 24.0)


def read_pose_columns(rows: Sequence[dict[str, str]], suffix: str = "") -> list[Pose | None]:
    """Poses from the raw columns (suffix "") or the smoothed ones (suffix "_smooth"); None where NaN."""
    poses = []
    for row in rows:
        values = np.array([float(row[f"{c}{suffix}"]) for c in ("tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw")])
        poses.append(None if np.isnan(values).any() else Pose.from_quaternion(values[3:], values[:3]))
    return poses


_OFFSET = re.compile(r"raw camera position\s+([-\d.eE]+),\s*([-\d.eE]+),\s*([-\d.eE]+)\s*m subtracted")


@dataclass
class HandJoints:
    """Skeletons in the raw camera frame (the CSV's origin shift undone), keyed by frame then hand."""

    skeletons: dict[int, dict[str, np.ndarray]]
    offset_found: bool


def read_joints(path: Path) -> HandJoints:
    offset = np.zeros(3)
    offset_found = False
    rows = []
    with path.open() as f:
        for line in f:
            if line.startswith("#"):
                if match := _OFFSET.search(line):
                    offset = np.array([float(v) for v in match.groups()])
                    offset_found = True
            else:
                rows.append(line)
    points: dict[int, dict[str, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for row in csv.DictReader(rows):
        points[int(row["frame"])][row["hand"]][int(row["joint"])] = np.array(
            [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]) + offset
    skeletons = {
        frame: {hand: np.array([joints[k] for k in sorted(joints)]) for hand, joints in hands.items()}
        for frame, hands in points.items()
    }
    return HandJoints(skeletons, offset_found)


def frame_count_problem(joints: HandJoints, video_frames: int) -> str | None:
    frames = sorted(joints.skeletons)
    if frames and frames[-1] >= video_frames:
        return f"the joints CSV has frame {frames[-1]} but the video only has {video_frames} frames"
    if len(frames) != video_frames:
        return f"the joints CSV has {len(frames)} frames, the video {video_frames}"
    return None
