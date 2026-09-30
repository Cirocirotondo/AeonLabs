"""extract-extrinsics: camera pose per video frame from the A4 reference paper."""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .debug_video import write_debug_video
from .focal import BOUNDS_PX_AT_832, estimate_focal
from .geometry import Intrinsics, paper_corners_world, stretched_16_9_aspect
from .io import frame_count_problem, load_intrinsics, read_joints, read_video, save_intrinsics, write_poses_csv
from .pipeline import FAILED, PAPER, FrameCorners, solve_poses, track_paper_corners
from .smoothing import smooth_poses


@dataclass
class ClipCorners:
    video: Path
    fps: float
    size: tuple[int, int]
    corners: list[FrameCorners]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="extract-extrinsics", description=__doc__)
    parser.add_argument("video", help="video file, or a glob pattern with --glob (quote it)")
    parser.add_argument("--glob", action="store_true", help="treat VIDEO as a glob pattern and process every match")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--joints", type=Path,
                        help="hand pose estimator CSV, to check its frame count against the video (single clip). With --glob, "
                             "<name>_raw_joints.csv next to each <name>_overlay.mp4 is used when present")
    parser.add_argument("--intrinsics", type=Path, help="intrinsics file to reuse or create (default OUT/intrinsics.json)")
    parser.add_argument("--estimate-focal", action="store_true",
                        help="estimate the focal length again even if the intrinsics file exists")
    parser.add_argument("--focal-px", type=float, help="use this focal length instead of estimating it")
    parser.add_argument("--paper-size", type=float, nargs=2, default=(297.0, 210.0), metavar=("LONG_MM", "SHORT_MM"),
                        help="reference paper size in mm (default A4)")
    parser.add_argument("--pixel-aspect", default="auto",
                        help="fy / fx, or 'auto': the video is a 16:9 original stretched to its frame size "
                             "(1.026 for 832x480, 1.0 for true 16:9)")
    parser.add_argument("--smooth-sigma", type=float, default=2.0, help="smoothing width in frames")
    parser.add_argument("--max-gap", type=int, default=12, help="longest gap (frames) the smoothed poses fill")
    parser.add_argument("--max-failed-fraction", type=float, default=0.2,
                        help="exit non-zero when a clip has more failed frames than this")
    parser.add_argument("--no-debug-video", action="store_true")
    return parser.parse_args(argv)


def _joints_for(video: Path, args: argparse.Namespace) -> Path | None:
    if not args.glob:
        return args.joints
    candidate = video.with_name(video.stem.removesuffix("_overlay") + "_raw_joints.csv")
    return candidate if candidate.exists() else None


def _resolve_intrinsics(clips: list[ClipCorners], args: argparse.Namespace, paper: np.ndarray) -> tuple[Intrinsics, str]:
    sizes = {c.size for c in clips}
    if len(sizes) > 1:
        sys.exit(f"all clips must have the same resolution to share intrinsics, got {sorted(sizes)}")
    width, height = sizes.pop()
    pixel_aspect = stretched_16_9_aspect(width, height) if args.pixel_aspect == "auto" else float(args.pixel_aspect)
    path = args.intrinsics or args.out / "intrinsics.json"
    names = [c.video.name for c in clips]

    if args.focal_px is not None:
        intrinsics = Intrinsics.centred(args.focal_px, width, height, pixel_aspect)
        save_intrinsics(path, intrinsics, None, "given with --focal-px", names)
        return intrinsics, f"focal {args.focal_px:.1f} px given with --focal-px"

    if path.exists() and not args.estimate_focal:
        saved = load_intrinsics(path)
        if (saved.intrinsics.width, saved.intrinsics.height) != (width, height):
            sys.exit(f"{path} is for {saved.intrinsics.width}x{saved.intrinsics.height} video, these clips are "
                     f"{width}x{height}; re-run with --estimate-focal")
        return saved.intrinsics, f"focal {saved.intrinsics.fx:.1f} ± {saved.focal_sigma_px:.1f} px, {saved.source}"

    measured = [f.corners for c in clips for f in c.corners if f.source == PAPER]
    estimate = estimate_focal(measured, (width, height), pixel_aspect, paper)
    intrinsics = Intrinsics.centred(estimate.focal_px, width, height, pixel_aspect)
    save_intrinsics(path, intrinsics, estimate, "estimated from the reference paper", names)
    print(f"focal length {estimate.focal_px:.1f} ± {estimate.sigma_px:.1f} px from {estimate.frames} frames "
          f"of {len(clips)} clip(s), corner noise {estimate.corner_noise_px:.2f} px -> {path}")
    if estimate.at_bound:
        print(f"WARNING: the focal length sits on the edge of the allowed range {BOUNDS_PX_AT_832} px (at 832 px "
              "wide); the paper barely constrains it, so its ± is not meaningful")
    return intrinsics, (f"focal {estimate.focal_px:.1f} ± {estimate.sigma_px:.1f} px estimated from the reference "
                        f"paper in {estimate.frames} frames, saved to {path.name}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    videos = sorted(Path(p) for p in glob.glob(args.video)) if args.glob else [Path(args.video)]
    if not videos:
        sys.exit(f"no video matches {args.video}")

    clips = []
    for video in videos:
        frames, fps = read_video(video)
        corners = track_paper_corners(frames)
        clips.append(ClipCorners(video, fps, (frames[0].shape[1], frames[0].shape[0]), corners))
        del frames
    paper = paper_corners_world(args.paper_size[0] / 1000, args.paper_size[1] / 1000)
    intrinsics, focal_line = _resolve_intrinsics(clips, args, paper)

    too_many_failures = False
    for clip in clips:
        poses = solve_poses(clip.corners, intrinsics, paper)
        smoothed = smooth_poses(poses, args.smooth_sigma, args.max_gap)
        stem = clip.video.stem
        extra = []

        joints_path = _joints_for(clip.video, args)
        if joints_path is not None:
            if problem := frame_count_problem(read_joints(joints_path), len(poses)):
                print(f"{stem}: WARNING {problem}")
                extra.append(f"frame count mismatch: {problem}")

        csv_path = args.out / f"{stem}_camera_poses.csv"
        write_poses_csv(csv_path, stem, clip.fps, poses, smoothed, intrinsics, focal_line,
                        f"smoothed columns: local linear fit with a Gaussian of {args.smooth_sigma:g} frames, "
                        f"gaps up to {args.max_gap} frames filled",
                        [f"reference paper: {args.paper_size[0]:g} x {args.paper_size[1]:g} mm", *extra])
        if not args.no_debug_video:
            frames, _ = read_video(clip.video)
            write_debug_video(args.out / f"{stem}_camera_debug.mp4", frames, clip.fps, clip.corners, poses,
                              smoothed, intrinsics, paper)

        counts = {s: sum(p.source == s for p in poses) for s in ("paper", "tracked", FAILED)}
        failed_fraction = counts[FAILED] / len(poses)
        measured = [p.reprojection_px for p in poses if p.source == PAPER]
        worst = f", worst paper reprojection {max(measured):.2f} px" if measured else ""
        print(f"{stem}: {len(poses)} frames, {counts['paper']} paper, {counts['tracked']} tracked, "
              f"{counts[FAILED]} failed{worst} -> {csv_path}")
        if failed_fraction > args.max_failed_fraction:
            print(f"{stem}: ERROR {100 * failed_fraction:.0f}% of frames failed "
                  f"(limit {100 * args.max_failed_fraction:.0f}%)")
            too_many_failures = True
    return 1 if too_many_failures else 0
