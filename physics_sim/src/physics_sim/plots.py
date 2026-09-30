"""Actual vs commanded joint angles after a physics replay: one figure for the arm, one for the hand."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .scene import ARM_JOINTS, HAND_JOINTS

# Two categorical slots of the reference palette (blue, orange), on its light surface.
ACTUAL_COLOR, TARGET_COLOR = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


def _panel(ax, t, actual, target, title):
    ax.plot(t, np.degrees(target), color=TARGET_COLOR, linewidth=1.4, linestyle=(0, (4, 2)), label="target")
    ax.plot(t, np.degrees(actual), color=ACTUAL_COLOR, linewidth=1.6, label="actual")
    error = np.degrees(np.abs(actual - target)).max()
    ax.set_title(f"{title}   max error {error:.1f}°", loc="left")


def joint_figures(t: np.ndarray, q: np.ndarray, target: np.ndarray, name: str):
    """``q`` and ``target``: (T, 26) arm then hand joint angles in radians. Returns (arm figure, hand figure)."""
    n_arm = len(ARM_JOINTS)

    arm_fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, num=f"Arm: actual vs target ({name})")
    for j, ax in enumerate(axes.flat):
        _panel(ax, t, q[:, j], target[:, j], ARM_JOINTS[j].removesuffix("_joint"))
    for ax in axes[:, 0]:
        ax.set_ylabel("angle (°)")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")

    hand_fig, axes = plt.subplots(5, 4, figsize=(13, 11), sharex=True, num=f"Hand: actual vs target ({name})")
    for k, ax in enumerate(axes.flat):
        finger, joint = divmod(k, 4)
        _panel(ax, t, q[:, n_arm + k], target[:, n_arm + k], f"{FINGERS[finger]} j{joint + 1}  ({HAND_JOINTS[k]})")
    for ax in axes[:, 0]:
        ax.set_ylabel("angle (°)")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")

    for fig, title in ((arm_fig, "UR5e arm"), (hand_fig, "DG5F hand")):
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="upper right", ncols=2)
        fig.suptitle(f"{title}: actual vs target joint angles, {name}", x=0.01, ha="left", color=TEXT, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    return arm_fig, hand_fig


def tube_figure(demo, tube_path: np.ndarray):
    """Height of the tube over time, and its path seen from above against the human grasp's."""
    t = demo.timestamp[: len(tube_path)]
    fig, (height, top) = plt.subplots(1, 2, figsize=(12, 4.8), num=f"Tube ({demo.name})")

    height.plot(t, 100 * (tube_path[:, 2] - tube_path[0, 2]), color=ACTUAL_COLOR, linewidth=1.6, label="tube")
    if demo.grasp_time is not None:
        height.axvline(demo.grasp_time, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
        height.annotate("human grasp", (demo.grasp_time, 0), xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=TEXT_SECONDARY)
    height.set_title("tube height above its start", loc="left")
    height.set_xlabel("time (s)")
    height.set_ylabel("height (cm)")

    human = demo.grasp_centre_path[: len(tube_path)]
    top.plot(100 * human[:, 0], 100 * human[:, 1], color=TARGET_COLOR, linewidth=1.4, linestyle=(0, (4, 2)),
             label="human grasp")
    top.plot(100 * tube_path[:, 0], 100 * tube_path[:, 1], color=ACTUAL_COLOR, linewidth=1.6, label="tube")
    for point, label in ((tube_path[0], "start"), (tube_path[-1], "end"), (demo.tube_goal, "human leaves it")):
        top.plot(100 * point[0], 100 * point[1], "o", color=TEXT, markersize=5)
        top.annotate(label, (100 * point[0], 100 * point[1]), xytext=(5, 5), textcoords="offset points",
                     fontsize=7, color=TEXT_SECONDARY)
    top.set_aspect("equal", adjustable="datalim")
    top.set_title("path seen from above", loc="left")
    top.set_xlabel("x (cm)")
    top.set_ylabel("y (cm)")

    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper right", ncols=2)
    fig.suptitle(f"Tube: {demo.name}", x=0.01, ha="left", color=TEXT, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
