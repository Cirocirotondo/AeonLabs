"""Replay a robot demonstration on the simulated robot, with physics, and report what went wrong.

The joint targets are sent to the position drives at 60 Hz, as the real robot would get
them; the drives, contacts and gravity decide where the robot actually goes.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from .scene import ARM_JOINTS, DEFAULT_TABLE_TOP_Z, JOINTS, Scene, Tube, _finger, build_scene, set_joints

SETTLE_SECONDS = 0.5
SATURATED = 0.98  # fraction of the effort limit counted as saturated
LIFTED = 0.02  # m above its starting height: the tube counts as lifted
TEST_TABLE_RAISE = 0.02  # m, --test
FINGER_NAMES = {"1": "thumb", "2": "index", "3": "middle", "4": "ring", "5": "pinky"}


@dataclass
class RobotDemonstration:
    name: str
    timestamp: np.ndarray  # (T,)
    q: np.ndarray  # (T, 26) arm then hand
    table_top_z: float
    tube: Tube | None = None
    tube_goal: np.ndarray | None = None  # where the human leaves the tube
    grasp_centre_path: np.ndarray | None = None  # (T, 3) the human grasp that carries it
    grasp_time: float | None = None

    @classmethod
    def load(cls, path: Path) -> "RobotDemonstration":
        d = np.load(path)
        hand = d["hand_q_measured"] if "hand_q_measured" in d.files else d["hand_q"]
        table = float(d["table_top_z"]) if "table_top_z" in d.files else DEFAULT_TABLE_TOP_Z
        demo = cls(Path(path).stem, d["timestamp"], np.hstack([d["arm_q"], hand]), table)
        if "tube_position" in d.files:
            demo.tube = Tube(
                length=float(d["tube_length"]), radius=float(d["tube_radius"]), mass=float(d["tube_mass"]),
                friction=float(d["tube_friction"]), position=tuple(d["tube_position"]), yaw=float(d["tube_yaw"]),
            )
            demo.tube_goal, demo.grasp_centre_path = d["tube_goal"], d["grasp_centre_path"]
            demo.grasp_time = float(d["tube_grasp_time"])
        return demo


@dataclass
class Log:
    """What happened, per control step (q, targets) and per contact category (worst values)."""

    q: list = field(default_factory=list)
    target: list = field(default_factory=list)
    saturated_steps: np.ndarray = field(default_factory=lambda: np.zeros(len(JOINTS)))
    substeps: int = 0
    tube_centre: list = field(default_factory=list)  # per control step, world
    # category -> {pair name -> [steps in contact, deepest penetration m, largest normal force N]}
    contacts: dict = field(default_factory=lambda: defaultdict(dict))


def _category(scene: Scene, b1: int, b2: int) -> str | None:
    table = int(scene.model.geom_bodyid[scene.table_geom])
    if table in (b1, b2):
        other = b2 if b1 == table else b1
        if other in scene.tube_bodies:
            return None  # the tube lies on the table: not news
        return "hand-table" if other in scene.hand_bodies else "arm-table"
    if (b1 in scene.tube_bodies) != (b2 in scene.tube_bodies):
        other = b2 if b1 in scene.tube_bodies else b1
        return "hand-tube" if other in scene.hand_bodies else "arm-tube"
    if b1 in scene.hand_bodies and b2 in scene.hand_bodies:
        f1, f2 = _finger(scene.body_name(b1)), _finger(scene.body_name(b2))
        return "finger-finger" if f1 and f2 and f1 != f2 else "hand-self"
    return None


def _record_contacts(scene: Scene, log: Log, step_seen: set) -> None:
    force = np.zeros(6)
    for i, c in enumerate(scene.data.contact[: scene.data.ncon]):
        b1, b2 = int(scene.model.geom_bodyid[c.geom1]), int(scene.model.geom_bodyid[c.geom2])
        category = _category(scene, b1, b2)
        if category is None:
            continue
        if category in ("hand-tube", "arm-tube"):  # the tube's segments are one object
            name = scene.body_name(b1 if b2 in scene.tube_bodies else b2)
        else:
            name = " / ".join(sorted((scene.body_name(b1), scene.body_name(b2))))
        mujoco.mj_contactForce(scene.model, scene.data, i, force)
        entry = log.contacts[category].setdefault(name, [0, 0.0, 0.0])
        if (category, name) not in step_seen:
            entry[0] += 1
            step_seen.add((category, name))
        entry[1] = max(entry[1], -c.dist)
        entry[2] = max(entry[2], abs(force[0]))


def replay(scene: Scene, demo: RobotDemonstration, viewer=None, realtime: bool = False, speed: float = 1.0) -> Log:
    """``speed`` only paces the viewer (0.5 = half speed); the physics is the same at any speed."""
    log = Log()
    model, data = scene.model, scene.data
    limits = np.abs(model.actuator_forcerange[scene.actuators, 1])
    mujoco.mj_resetData(model, data)  # the tube back where it starts
    set_joints(scene, demo.q[0])
    targets = [demo.q[0]] * int(round(SETTLE_SECONDS / scene.physics.control_dt)) + list(demo.q)
    settle_steps = len(targets) - len(demo.q)
    for step, target in enumerate(targets):
        start = time.perf_counter()
        data.ctrl[scene.actuators] = target
        if scene.ghost_qpos_adr is not None:
            data.qpos[scene.ghost_qpos_adr] = target
            data.qvel[[model.joint("ghost_" + n).dofadr[0] for n in JOINTS]] = 0.0
        seen: set = set()
        for _ in range(scene.physics.substeps):
            mujoco.mj_step(model, data)
            if step >= settle_steps:
                log.substeps += 1
                log.saturated_steps += np.abs(data.actuator_force[scene.actuators]) >= SATURATED * limits
                _record_contacts(scene, log, seen)
        if step >= settle_steps:
            if scene.tube_bodies:
                log.tube_centre.append(scene.tube_centre())
            log.q.append(data.qpos[scene.qpos_adr].copy())
            log.target.append(np.asarray(target).copy())
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
        if realtime:
            time.sleep(max(0.0, scene.physics.control_dt / speed - (time.perf_counter() - start)))
    return log


def tube_report(log: Log, demo: RobotDemonstration) -> str:
    if not log.tube_centre:
        return ""
    path = np.array(log.tube_centre)
    t = demo.timestamp[: len(path)]
    lift = path[:, 2] - path[0, 2]
    lifted = lift > LIFTED
    lines = [
        f"Tube: lifted up to {100 * lift.max():.1f} cm, above {100 * LIFTED:.0f} cm for {lifted.mean() * t[-1]:.2f} s "
        f"of {t[-1]:.2f} s" + (f" (from {t[lifted][0]:.2f} s)" if lifted.any() else ""),
    ]
    if lifted.any() and lift[-1] < 0.005 and lifted[-len(lifted) // 4 :].sum() == 0:
        dropped = t[lifted][-1]
        lines.append(f"  back on the table from {dropped:.2f} s")
    goal = demo.tube_goal
    moved = np.linalg.norm(path[-1, :2] - path[0, :2])
    lines.append(
        f"  ends {100 * np.linalg.norm(path[-1, :2] - goal[:2]):.1f} cm from where the human leaves it "
        f"(moved {100 * moved:.1f} cm of the human's {100 * np.linalg.norm(goal[:2] - path[0, :2]):.1f} cm)"
    )
    touching = defaultdict(lambda: [0, 0.0])
    for name, (count, _, force) in log.contacts.get("hand-tube", {}).items():
        finger = FINGER_NAMES.get(_finger(name) or "", "palm")
        touching[finger][0] = max(touching[finger][0], count)
        touching[finger][1] = max(touching[finger][1], force)
    steps = len(log.q)
    lines.append(
        "  touched by: "
        + (", ".join(f"{f} {c}/{steps} steps (up to {force:.1f} N)" for f, (c, force) in touching.items()) or "nothing")
    )
    return "\n".join(lines)


def report(log: Log) -> str:
    q, target = np.array(log.q), np.array(log.target)
    error = np.degrees(np.abs(q - target))
    arm, hand = error[:, : len(ARM_JOINTS)], error[:, len(ARM_JOINTS) :]
    worst = np.argsort(-error.max(0))[:5]
    saturated = log.saturated_steps / max(log.substeps, 1)
    lines = [
        f"Tracking error (deg): arm max {arm.max():.1f}, rms {np.sqrt((arm**2).mean()):.2f}; "
        f"hand max {hand.max():.1f}, rms {np.sqrt((hand**2).mean()):.2f}",
        "  worst joints: " + ", ".join(f"{JOINTS[j]} {error[:, j].max():.1f}" for j in worst),
        "Drives at their effort limit: "
        + (", ".join(f"{JOINTS[j]} {100 * saturated[j]:.0f}% of the time" for j in np.flatnonzero(saturated > 0.01)) or "none"),
    ]
    steps = len(log.q)
    for category in ("hand-table", "arm-table", "arm-tube", "finger-finger", "hand-self"):
        pairs = log.contacts.get(category, {})
        if not pairs:
            lines.append(f"Contacts {category}: none")
            continue
        lines.append(f"Contacts {category}:")
        for name, (count, depth, force) in sorted(pairs.items(), key=lambda kv: -kv[1][1]):
            lines.append(
                f"  {name}: {count}/{steps} steps, deepest {1000 * depth:.1f} mm, largest force {force:.1f} N"
            )
    return "\n".join(lines)


def show_joint_plots(demo: RobotDemonstration, log: Log, save_dir: Path | None = None):
    """Open (or save) the arm and hand figures of actual vs commanded joint angles."""
    import matplotlib.pyplot as plt

    from .plots import joint_figures, tube_figure

    q, target = np.array(log.q), np.array(log.target)
    figures = joint_figures(demo.timestamp[: len(q)], q, target, demo.name)
    if log.tube_centre:
        figures = figures + (tube_figure(demo, np.array(log.tube_centre)),)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for fig, part in zip(figures, ("arm", "hand", "tube")):
            path = save_dir / f"{demo.name}_{part}{'' if part == 'tube' else '_joints'}.png"
            fig.savefig(path, dpi=110)
            print(f"Saved {path}")
        plt.close("all")
    else:
        plt.show(block=False)
        plt.pause(0.1)


def _figures_open() -> bool:
    import matplotlib.pyplot as plt

    return bool(plt.get_fignums())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demonstration", type=Path, help="a robot demonstration .npz (hand_retarget's export)")
    parser.add_argument("--headless", action="store_true", help="no viewer: replay once as fast as possible")
    parser.add_argument("--loop", action="store_true", help="in the viewer, replay again until it is closed")
    parser.add_argument("--no-ghost", dest="ghost", action="store_false", help="hide the commanded-pose ghost")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer playback speed, e.g. 0.25 for a quarter")
    parser.add_argument(
        "--test", action="store_true",
        help=f"raise the table (and the tube on it) by {100 * TEST_TABLE_RAISE:.0f} cm and let the hand pass through the table",
    )
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="skip the joint-angle figures")
    parser.add_argument("--save-plots", type=Path, default=None, help="with --headless: save the figures here")
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be positive")
    demo = RobotDemonstration.load(args.demonstration)
    if args.test:
        demo.table_top_z += TEST_TABLE_RAISE
        if demo.tube is not None:
            demo.tube = replace(demo.tube, position=tuple(np.asarray(demo.tube.position) + (0.0, 0.0, TEST_TABLE_RAISE)))
            demo.tube_goal = demo.tube_goal + (0.0, 0.0, TEST_TABLE_RAISE)
    scene = build_scene(
        demo.table_top_z, with_ghost=args.ghost, rest_poses=(demo.q[0],), tube=demo.tube,
        hand_table_contact=not args.test,
    )
    print(f"{demo.name}: {len(demo.q)} steps at 60 Hz, table top at {demo.table_top_z:.3f} m")
    if args.test:
        print(f"Test mode: table and tube raised by {100 * TEST_TABLE_RAISE:.0f} cm, hand-table contacts off")
    if args.headless:
        log = replay(scene, demo)
        print(report(log))
        print(tube_report(log, demo))
        if args.save_plots:
            show_joint_plots(demo, log, args.save_plots)
        return
    import matplotlib.pyplot as plt
    import mujoco.viewer

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = (0.4, 0.0, 0.6)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 60.0, -25.0
        while viewer.is_running():
            log = replay(scene, demo, viewer, realtime=True, speed=args.speed)
            print(report(log))
            print(tube_report(log, demo))
            if args.plots and len(log.q):
                plt.close("all")
                show_joint_plots(demo, log)
            if not args.loop:
                break
        # Keep the figures and the viewer responsive until both are closed.
        while viewer.is_running() or _figures_open():
            if _figures_open():
                plt.pause(0.1)
            else:
                time.sleep(0.1)
