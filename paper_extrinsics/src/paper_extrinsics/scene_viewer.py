"""view-scene: interactive 3D view of a human demonstration in the paper frame.

Shows:
  * the paper frame (origin at the centre of the reference paper, x toward the top of the first
    frame, z up), the reference paper and the table area the camera sees during the clip
  * the camera of the current frame: its axes (OpenCV: x right, y down, z optical axis), its real
    frustum from the saved intrinsics down to the table, and the footprint of the image on the table
  * the camera trajectory over the whole clip
  * the hand skeleton of the current frame in the paper frame, coloured by finger, and the wrist
    trajectory
  * the video frame, with the paper outline projected back from the camera pose

The skeleton is shown exactly as the hand pose estimator gives it (metric, in the camera frame),
moved into the paper frame with each frame's camera pose; nothing about the hand is adjusted.

Usage:
  view-scene CLIP_overlay.mp4 --out OUT_DIR [--joints CLIP_raw_joints.csv] [--save out.gif | --png out.png]

Keys: space = play/pause, left/right = step one frame, s = raw/smoothed camera.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button, CheckButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import Intrinsics, Pose, paper_corners_world, project
from .io import HandJoints, read_joints, read_pose_columns, read_poses_csv, read_poses_header, read_video

# dpl/MANO 21-joint order: 0 wrist | 1-3 index | 4-6 middle | 7-9 pinky | 10-12 ring | 13-15 thumb | 16-20 tips
FINGERS = {
    "thumb": ([0, 13, 14, 15, 20], "#d62728"),
    "index": ([0, 1, 2, 3, 16], "#ff7f0e"),
    "middle": ([0, 4, 5, 6, 17], "#2ca02c"),
    "ring": ([0, 10, 11, 12, 19], "#1f77b4"),
    "pinky": ([0, 7, 8, 9, 18], "#9467bd"),
}
PALM = [1, 4, 10, 7]
AXIS_COLOURS = ("#e41a1c", "#4daf4a", "#377eb8")
TABLE_COLOUR = "#c9b98f"
TABLE_MARGIN_M = 0.05


def image_footprint(intrinsics: Intrinsics, pose: Pose) -> np.ndarray | None:
    """Where the four image corners' rays hit the table (z = 0), or None if a ray misses it."""
    pixels = np.array([[0, 0], [intrinsics.width, 0], [intrinsics.width, intrinsics.height], [0, intrinsics.height]], float)
    rays_cam = np.column_stack([(pixels[:, 0] - intrinsics.cx) / intrinsics.fx,
                                (pixels[:, 1] - intrinsics.cy) / intrinsics.fy, np.ones(4)])
    rays = rays_cam @ pose.rotation.T
    if np.any(rays[:, 2] >= 0):
        return None
    return pose.translation + (-pose.translation[2] / rays[:, 2])[:, None] * rays


def _hand_arrays(joints: HandJoints, n: int) -> dict[str, np.ndarray]:
    hands: dict[str, np.ndarray] = {}
    for frame, per_hand in joints.skeletons.items():
        for hand, points in per_hand.items():
            if frame < n:
                hands.setdefault(hand, np.full((n, 21, 3), np.nan))[frame] = points
    return hands


def _set_frame_axes(lines, origin, rotation, length):
    for line, axis in zip(lines, rotation.T):
        line.set_data_3d(*np.array([origin, origin + length * axis]).T)


class SceneViewer:
    def __init__(self, video: Path, poses_csv: Path, joints_csv: Path | None):
        rows = read_poses_csv(poses_csv)
        header = read_poses_header(poses_csv)
        self.intrinsics, self.fps = header.intrinsics, header.fps
        self.paper = paper_corners_world(*header.paper_size_m)
        self.sources = [r["source"] for r in rows]
        self.reprojection = [float(r["reprojection_px"]) for r in rows]
        self.camera = {"raw": read_pose_columns(rows), "smoothed": read_pose_columns(rows, "_smooth")}
        self.n = len(rows)
        frames, _ = read_video(video)
        self.images = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[: self.n]]
        self.clip = video.stem

        self.hands_cam = _hand_arrays(read_joints(joints_csv), self.n) if joints_csv else {}
        self.camera_mode, self.playing, self.frame = "smoothed", False, 0
        self.hands_world = {key: self._hands_in_world(key) for key in ("raw", "smoothed")}
        self._build_figure(video.name)
        self.update(0)

    # ---- data
    def _hands_in_world(self, camera_mode: str) -> dict[str, np.ndarray]:
        out = {}
        for hand, points in self.hands_cam.items():
            world = np.full_like(points, np.nan)
            for i, pose in enumerate(self.camera[camera_mode]):
                if pose is not None and np.isfinite(points[i]).all():
                    world[i] = pose.apply(points[i])
            out[hand] = world
        return out

    # ---- figure
    def _build_figure(self, title: str) -> None:
        self.fig = plt.figure(figsize=(15, 8.5))
        ax = self.ax = self.fig.add_axes([0.0, 0.12, 0.58, 0.86], projection="3d")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z up [m]")
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=24, azim=-140)

        footprints = [f for p in self.camera["smoothed"] + self.camera["raw"] if p is not None
                      and (f := image_footprint(self.intrinsics, p)) is not None]
        seen = np.concatenate(footprints + [self.paper])
        lo, hi = seen[:, :2].min(0) - TABLE_MARGIN_M, seen[:, :2].max(0) + TABLE_MARGIN_M
        table = np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])
        ax.add_collection3d(Poly3DCollection([table], facecolor=TABLE_COLOUR, alpha=0.35, edgecolor="0.5"))
        ax.add_collection3d(Poly3DCollection([self.paper], facecolor="white", edgecolor="0.2", alpha=0.95))
        for k, corner in enumerate(self.paper):
            ax.text(*corner, f"C{k}", fontsize=7, color="0.3")
        for axis, colour, name in zip(np.eye(3), AXIS_COLOURS, "xyz"):
            ax.plot(*np.array([np.zeros(3), 0.08 * axis]).T, color=colour, lw=2.5)
            ax.text(*(0.09 * axis), name, color=colour, fontsize=8)

        centres = np.array([p.translation for p in self.camera["smoothed"] if p is not None])
        (self.camera_path,) = ax.plot(*centres.T, color="0.45", lw=1)
        self.camera_axes = [ax.plot([], [], [], color=c, lw=2)[0] for c in AXIS_COLOURS]
        self.frustum_edges = [ax.plot([], [], [], color="0.35", lw=0.8)[0] for _ in range(4)]
        (self.footprint,) = ax.plot([], [], [], color="0.35", lw=1)
        (self.optical_axis,) = ax.plot([], [], [], color="0.5", lw=0.8, ls=":")
        self.camera_dot = ax.scatter([], [], [], color="k", s=30, depthshade=False)

        self.bones, self.palms, self.wrist_paths = {}, {}, {}
        for n, hand in enumerate(self.hands_cam):
            for finger, (chain, colour) in FINGERS.items():
                self.bones[hand, finger] = ax.plot([], [], [], color=colour, lw=2, marker="o", ms=3,
                                                   label=finger if n == 0 else None)[0]
            self.palms[hand] = ax.plot([], [], [], color="0.4", lw=1.2)[0]
            self.wrist_paths[hand] = ax.plot([], [], [], color="#8c564b", lw=1, alpha=0.8)[0]
        if self.hands_cam:
            ax.legend(loc="upper left", fontsize=8)

        everything = [table, centres] + [w.reshape(-1, 3) for w in self.hands_world["smoothed"].values()]
        points = np.concatenate(everything)
        points = points[np.isfinite(points).all(axis=1)]
        middle, half = (points.min(0) + points.max(0)) / 2, (points.max(0) - points.min(0)).max() / 2 * 1.05
        ax.set_xlim(middle[0] - half, middle[0] + half)
        ax.set_ylim(middle[1] - half, middle[1] + half)
        ax.set_zlim(max(-0.02, middle[2] - half), max(-0.02, middle[2] - half) + 2 * half)
        ax.set_box_aspect((1, 1, 1))

        self.image_ax = self.fig.add_axes([0.59, 0.42, 0.40, 0.55])
        self.image_ax.set_axis_off()
        self.image = self.image_ax.imshow(self.images[0])
        (self.image_outline,) = self.image_ax.plot([], [], color="#00e5ff", lw=1)
        self.info = self.fig.text(0.60, 0.40, "", fontsize=8.5, va="top", family="monospace")

        self.slider = Slider(self.fig.add_axes([0.08, 0.05, 0.45, 0.03]), "frame", 0, self.n - 1, valinit=0, valstep=1)
        self.slider.on_changed(lambda v: self.update(int(v)))
        self.button = Button(self.fig.add_axes([0.56, 0.04, 0.07, 0.05]), "Play")
        self.button.on_clicked(lambda _: self.toggle_play())
        self.checks = CheckButtons(self.fig.add_axes([0.65, 0.02, 0.16, 0.09]),
                                   ["raw camera", "trajectories"], [False, True])
        self.checks.on_clicked(self._on_check)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.animation = FuncAnimation(self.fig, self._tick, interval=1000 / self.fps, cache_frame_data=False)
        self.animation.event_source.stop()

    # ---- interaction
    def _on_check(self, label: str) -> None:
        if label == "raw camera":
            self.camera_mode = "raw" if self.camera_mode == "smoothed" else "smoothed"
        self.update(self.frame)

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.toggle_play()
        elif event.key in ("left", "right"):
            self.slider.set_val((self.frame + (1 if event.key == "right" else -1)) % self.n)
        elif event.key == "s":
            self.checks.set_active(0)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.button.label.set_text("Pause" if self.playing else "Play")
        (self.animation.event_source.start if self.playing else self.animation.event_source.stop)()
        self.fig.canvas.draw_idle()

    def _tick(self, _) -> None:
        if self.playing:
            self.slider.set_val((self.frame + 1) % self.n)

    # ---- rendering
    def update(self, i: int) -> None:
        self.frame = i
        pose = self.camera[self.camera_mode][i]
        show_paths = self.checks.get_status()[1]
        self.camera_path.set_visible(show_paths)

        footprint = image_footprint(self.intrinsics, pose) if pose is not None else None
        empty = np.empty((3, 0))
        if pose is None:
            for line in self.camera_axes + self.frustum_edges + [self.footprint, self.optical_axis]:
                line.set_data_3d(*empty)
            self.camera_dot._offsets3d = empty
            self.image_outline.set_data([], [])
        else:
            centre = pose.translation
            _set_frame_axes(self.camera_axes, centre, pose.rotation, 0.06)
            for line, corner in zip(self.frustum_edges, footprint if footprint is not None else []):
                line.set_data_3d(*np.array([centre, corner]).T)
            ring = np.vstack([footprint, footprint[:1]]) if footprint is not None else empty.T
            self.footprint.set_data_3d(*ring.T)
            axis = pose.rotation[:, 2]
            hit = centre + (-centre[2] / axis[2]) * axis if axis[2] < 0 else centre
            self.optical_axis.set_data_3d(*np.array([centre, hit]).T)
            self.camera_dot._offsets3d = tuple(centre[:, None])
            outline = project(self.intrinsics, pose, self.paper)
            self.image_outline.set_data(*np.vstack([outline, outline[:1]]).T)

        lines = [f"clip    {self.clip}", f"frame   {i} / {self.n - 1}   t = {i / self.fps:.3f} s",
                 f"source  {self.sources[i]}   reprojection {self.reprojection[i]:.2f} px",
                 f"camera  {self.camera_mode}"]
        if pose is not None:
            tilt = np.degrees(np.arccos(np.clip(-pose.rotation[2, 2], -1, 1)))
            lines += [f"  position  x {pose.translation[0]:+.3f}  y {pose.translation[1]:+.3f}  "
                      f"z {pose.translation[2]:+.3f} m", f"  tilt from vertical {tilt:.1f} deg"]

        world = self.hands_world[self.camera_mode]
        for hand in self.hands_cam:
            points = world[hand]
            current = points[i]
            present = np.isfinite(current).all()
            for finger, (chain, _) in FINGERS.items():
                self.bones[hand, finger].set_data_3d(*(current[chain].T if present else empty))
            self.palms[hand].set_data_3d(*(current[PALM].T if present else empty))
            wrists = points[:, 0]
            self.wrist_paths[hand].set_data_3d(*(wrists.T if show_paths else empty))
            if present:
                lines.append(f"  {hand} wrist  x {current[0, 0]:+.3f}  y {current[0, 1]:+.3f}  "
                             f"z {current[0, 2]:+.3f} m, lowest keypoint {100 * current[:, 2].min():.1f} cm")
            else:
                lines.append(f"  {hand}: absent")
        self.image.set_data(self.images[i])
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="view-scene", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="the clip's video")
    parser.add_argument("--out", type=Path, required=True, help="extract-extrinsics output directory")
    parser.add_argument("--joints", type=Path,
                        help="hand pose estimator CSV (default: <name>_raw_joints.csv next to <name>_overlay.mp4)")
    parser.add_argument("--save", metavar="OUT.gif", type=Path, help="render every frame to an animated GIF")
    parser.add_argument("--png", metavar="OUT.png", type=Path, help="save a still of --frame and exit")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args(argv)

    poses_csv = args.out / f"{args.video.stem}_camera_poses.csv"
    if not poses_csv.exists():
        parser.error(f"{poses_csv} not found; run extract-extrinsics on the clip first")
    joints = args.joints
    if joints is None:
        candidate = args.video.with_name(args.video.stem.removesuffix("_overlay") + "_raw_joints.csv")
        joints = candidate if candidate.exists() else None

    if args.save or args.png:
        plt.switch_backend("Agg")
    viewer = SceneViewer(args.video, poses_csv, joints)
    if args.png:
        viewer.slider.set_val(args.frame)
        viewer.fig.savefig(args.png, dpi=100)
    elif args.save:
        animation = FuncAnimation(viewer.fig, viewer.update, frames=viewer.n, interval=1000 / viewer.fps)
        animation.save(args.save, writer=PillowWriter(fps=viewer.fps))
        print("wrote", args.save)
    else:
        plt.show()
