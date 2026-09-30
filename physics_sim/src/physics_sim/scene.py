"""The MuJoCo world: UR5e + right DG5F on simtoolreal's table, with position drives and physics.

Physics, drives and contact parameters are simtoolreal_newton's MuJoCo sim2sim
(``simtoolreal_newton/sim2sim/mujoco_sim.py`` and ``cfg/``), so a replay here behaves as a
policy rollout there. One deliberate difference: simtoolreal switches robot-table
contacts and most hand self-contacts off for training; here they are on, because the
point is to find out what the real robot would hit.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# The robot's assets live in the sibling hand_retarget project (copied there from simtoolreal).
DEFAULT_ASSETS = Path(__file__).resolve().parents[3] / "hand_retarget" / "assets"
URDF_NAME = "ur5e_right_dg5f.urdf"

ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
HAND_JOINTS = tuple(f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5))
JOINTS = ARM_JOINTS + HAND_JOINTS
HAND_ROOT_BODY = "rl_dg_mount"  # every body below it belongs to the hand
FINGERTIP_BODIES = tuple(f"rl_dg_{finger}_{link}" for finger in range(1, 6) for link in ("4", "tip"))

# simtoolreal: robot base pose and table.
ROBOT_BASE_POSITION = np.array([0.0, 0.6, 0.55])
TABLE_SIZE = np.array([0.75, 0.75, 0.3])
DEFAULT_TABLE_TOP_Z = ROBOT_BASE_POSITION[2] - 0.035

# simtoolreal: envs/pd_gains.py, with cfg control.hand_stiffness_scale = 0.116.
ARM_STIFFNESS = (1000.0, 1000.0, 1000.0, 200.0, 200.0, 100.0)
ARM_DAMPING = (100.0, 100.0, 100.0, 10.0, 10.0, 10.0)
HAND_STIFFNESS_SCALE = 0.116
HAND_STIFFNESS = tuple(
    HAND_STIFFNESS_SCALE * k
    for k in (
        42.9718, 400.0, 42.9718, 42.9718,
        42.9718, 42.9718, 42.9718, 42.9718,
        42.9718, 42.9718, 42.9718, 42.9718,
        42.9718, 42.9718, 42.9718, 42.9718,
        42.9718, 42.9718, 42.9718, 42.9718,
    )
)
HAND_DAMPING = (
    0.1, 0.9475, 0.3012, 0.1821,
    0.7523, 0.4126, 0.2856, 0.1365,
    0.7587, 0.4126, 0.2856, 0.1365,
    0.7274, 0.4126, 0.2856, 0.1365,
    0.2662, 0.4796, 0.3012, 0.1821,
)

# Collision groups.
TABLE_BIT, HAND_BIT, ARM_BIT, TUBE_BIT = 1, 2, 4, 8


@dataclass(frozen=True)
class Tube:
    """A rubber tube lying on the table: a chain of capsules joined by springy ball joints,
    so it bends and twists a little but does not squash."""

    length: float  # m, end to end
    radius: float
    mass: float  # kg
    friction: float
    position: tuple  # world centre at the start
    yaw: float  # rad, heading of its axis
    segments: int = 10
    # Per joint. With these, the 30 cm tube held by one end droops 1.1 cm at the other.
    bend_stiffness: float = 2.0  # N m / rad
    bend_damping: float = 0.01  # N m s / rad
    # Rotor inertia added to each joint so the stiff, light segments stay stable at 480 Hz.
    armature: float = 2e-5

    @property
    def segment_length(self) -> float:
        """Axis length of one capsule; the rounded ends make up the rest of the tube."""
        return (self.length - 2 * self.radius) / self.segments


@dataclass(frozen=True)
class Physics:
    """simtoolreal's cfg/base_config.py (sim, sim.mjwarp) and asset friction."""

    control_dt: float = 1.0 / 60.0
    substeps: int = 8
    integrator: int = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    cone: int = mujoco.mjtCone.mjCONE_ELLIPTIC
    impratio: float = 20.0
    iterations: int = 100
    ls_iterations: int = 50
    tolerance: float = 1e-6
    solref: tuple = (0.012, 1.4)
    solimp: tuple = (0.99, 0.999, 0.002, 0.5, 2.0)
    condim: int = 4
    friction: float = 0.5
    fingertip_friction: float = 1.0
    fingertip_torsional_friction: float = 0.1
    table_friction: float = 0.5


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_adr: np.ndarray  # (26,) robot joints
    qvel_adr: np.ndarray
    actuators: np.ndarray
    ghost_qpos_adr: np.ndarray | None
    hand_bodies: set[int]
    arm_bodies: set[int]
    table_geom: int
    excluded_hand_pairs: set[frozenset[str]]
    physics: Physics
    tube_bodies: set[int]

    def tube_centre(self) -> np.ndarray | None:
        """Centre of mass of the tube, world."""
        if not self.tube_bodies:
            return None
        bodies = sorted(self.tube_bodies)
        mass = self.model.body_mass[bodies]
        return (self.data.xipos[bodies] * mass[:, None]).sum(0) / mass.sum()

    def body_name(self, body: int) -> str:
        return self.model.body(body).name


def _mujoco_urdf(assets: Path, directory: Path) -> Path:
    """The URDF with absolute mesh paths, fixed links kept as bodies, visual meshes dropped
    (MuJoCo cannot read the .dae files; the collision meshes are drawn instead)."""
    text = (assets / URDF_NAME).read_text()
    text = re.sub(
        r'(<robot\s+name="[^"]+">)',
        r'\1\n  <mujoco><compiler strippath="false" fusestatic="false" discardvisual="true"/></mujoco>',
        text,
        count=1,
    )
    text = re.sub(r'filename="([^"]+)"', lambda m: f'filename="{(assets / m.group(1)).resolve()}"', text)
    path = directory / "robot_mujoco.urdf"
    path.write_text(text)
    return path


def _descendants(model: mujoco.MjModel, root: int) -> set[int]:
    out = set()
    for body in range(1, model.nbody):
        b = body
        while b != 0:
            if b == root:
                out.add(body)
                break
            b = int(model.body_parentid[b])
    return out


def _finger(name: str) -> str | None:
    parts = name.removeprefix("ghost_").split("_")
    return parts[2] if len(parts) >= 4 and parts[2].isdigit() else None


def build_scene(
    table_top_z: float = DEFAULT_TABLE_TOP_Z,
    with_ghost: bool = True,
    assets: Path = DEFAULT_ASSETS,
    physics: Physics = Physics(),
    rest_poses: tuple[np.ndarray, ...] = (),
    tube: Tube | None = None,
    hand_table_contact: bool = True,
) -> Scene:
    """Build and compile the world.

    ``rest_poses`` (26 joint angles each): hand links that already touch in these poses
    touch by design (neighbouring links whose meshes overlap at the joints) and are
    excluded from contact; the zero pose is always among them.
    """
    with tempfile.TemporaryDirectory(prefix="physics_sim_") as tmp:
        urdf = _mujoco_urdf(assets, Path(tmp))
        excluded: set[frozenset[str]] = set()
        for _ in range(2):  # first pass finds the design overlaps, second compiles without them
            spec = mujoco.MjSpec.from_file(str(urdf))
            if with_ghost:
                ghost = mujoco.MjSpec.from_file(str(urdf))
                spec.worldbody.add_frame().attach_body(ghost.worldbody.first_body(), prefix="ghost_")
            _add_world(spec, table_top_z)
            if tube is not None:
                _add_tube(spec, tube)
            _add_actuators(spec, urdf)
            for pair in excluded:
                a, b = sorted(pair)
                exclude = spec.add_exclude()
                exclude.bodyname1, exclude.bodyname2 = a, b
            # simtoolreal runs the robot gravity-compensated; the real UR5e and DG5F do too.
            for body in spec.bodies:
                if body.name not in ("world", "table") and not body.name.startswith("tube_"):
                    body.gravcomp = 1.0
            model = spec.compile()
            scene = _configure(model, table_top_z, with_ghost, physics, excluded, tube, hand_table_contact)
            if excluded or not _find_design_overlaps(scene, rest_poses):
                break
            excluded = _find_design_overlaps(scene, rest_poses)
        return scene


def _add_world(spec: mujoco.MjSpec, table_top_z: float) -> None:
    floor = spec.worldbody.add_geom()
    floor.name, floor.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([1.5, 1.5, 0.05])
    floor.rgba = np.array([0.20, 0.25, 0.28, 1.0])
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = np.array([0.0, 0.0, table_top_z - TABLE_SIZE[2] / 2])
    geom = table.add_geom()
    geom.name, geom.type = "table_geom", mujoco.mjtGeom.mjGEOM_BOX
    geom.size = TABLE_SIZE / 2
    geom.rgba = np.array([0.82, 0.56, 0.35, 1.0])
    light = spec.worldbody.add_light()
    light.pos, light.dir = np.array([0.0, -1.0, 1.5]), np.array([0.0, 0.5, -1.0])
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL


def _add_tube(spec: mujoco.MjSpec, tube: Tube) -> None:
    axis = np.array([np.cos(tube.yaw), np.sin(tube.yaw), 0.0])
    step = tube.segment_length
    start = np.asarray(tube.position) - axis * step * tube.segments / 2
    parent = spec.worldbody
    for i in range(tube.segments):
        body = parent.add_body()
        body.name = f"tube_{i}"
        if i == 0:
            body.pos = start
            body.quat = np.array([np.cos(tube.yaw / 2), 0.0, 0.0, np.sin(tube.yaw / 2)])
            free = body.add_joint()
            free.name, free.type = "tube_free", mujoco.mjtJoint.mjJNT_FREE
        else:
            body.pos = np.array([step, 0.0, 0.0])
            joint = body.add_joint()
            joint.name, joint.type = f"tube_bend_{i}", mujoco.mjtJoint.mjJNT_BALL
            # MuJoCo 3.14 takes stiffness and damping as polynomials; only the linear term is used.
            joint.stiffness = np.array([tube.bend_stiffness, 0.0, 0.0])
            joint.damping = np.array([tube.bend_damping, 0.0, 0.0])
            joint.armature = tube.armature
        geom = body.add_geom()
        geom.name, geom.type = f"tube_geom_{i}", mujoco.mjtGeom.mjGEOM_CAPSULE
        geom.fromto = np.array([0.0, 0.0, 0.0, step, 0.0, 0.0])
        geom.size = np.array([tube.radius, 0.0, 0.0])
        geom.mass = tube.mass / tube.segments
        geom.rgba = np.array([0.24, 0.75, 0.31, 1.0])
        parent = body


def _add_actuators(spec: mujoco.MjSpec, urdf: Path) -> None:
    import xml.etree.ElementTree as ET

    limits = {
        j.get("name"): j.find("limit")
        for j in ET.parse(urdf).getroot().findall("joint")
        if j.find("limit") is not None
    }
    for name, kp, kv in zip(JOINTS, ARM_STIFFNESS + HAND_STIFFNESS, ARM_DAMPING + HAND_DAMPING):
        limit = limits[name]
        actuator = spec.add_actuator()
        actuator.name, actuator.trntype, actuator.target = f"{name}_pos", mujoco.mjtTrn.mjTRN_JOINT, name
        actuator.ctrllimited = True
        actuator.ctrlrange = np.array([float(limit.get("lower")), float(limit.get("upper"))])
        actuator.forcelimited = True
        effort = float(limit.get("effort"))
        actuator.forcerange = np.array([-effort, effort])
        actuator.gaintype, actuator.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = kp
        actuator.biasprm[1], actuator.biasprm[2] = -kp, -kv


def _configure(
    model, table_top_z, with_ghost, physics: Physics, excluded, tube: Tube | None, hand_table_contact: bool = True
) -> Scene:
    hand_vs_table = HAND_BIT if hand_table_contact else 0
    opt = model.opt
    opt.timestep = physics.control_dt / physics.substeps
    opt.integrator, opt.cone, opt.impratio = physics.integrator, physics.cone, physics.impratio
    opt.iterations, opt.ls_iterations, opt.tolerance = physics.iterations, physics.ls_iterations, physics.tolerance

    base = model.body("base_link").id
    model.body_pos[base] = ROBOT_BASE_POSITION
    robot_bodies = _descendants(model, base)
    hand_bodies = _descendants(model, model.body(HAND_ROOT_BODY).id)
    arm_bodies = robot_bodies - hand_bodies
    ghost_bodies: set[int] = set()
    if with_ghost:
        ghost_base = model.body("ghost_base_link").id
        # The ghost shows the commanded pose beside the robot, as in simtoolreal's viewer.
        model.body_pos[ghost_base] = ROBOT_BASE_POSITION + np.array([0.8, 0.0, 0.0])
        ghost_bodies = _descendants(model, ghost_base)
    tips = {model.body(n).id for n in FINGERTIP_BODIES}
    table_geom = model.geom("table_geom").id
    tube_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith("tube_")}

    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        friction, torsional = physics.friction, 0.0
        if g == table_geom:
            contype, conaffinity, friction = TABLE_BIT, hand_vs_table | ARM_BIT | TUBE_BIT, physics.table_friction
        elif body in tube_bodies:
            # Not against itself: neighbouring capsules overlap by design.
            contype, conaffinity, friction = TUBE_BIT, TABLE_BIT | HAND_BIT | ARM_BIT, tube.friction
        elif body in hand_bodies:
            contype, conaffinity = HAND_BIT, (TABLE_BIT if hand_table_contact else 0) | HAND_BIT | TUBE_BIT
            if body in tips:
                friction, torsional = physics.fingertip_friction, physics.fingertip_torsional_friction
        elif body in arm_bodies:
            contype, conaffinity = ARM_BIT, TABLE_BIT | TUBE_BIT
        else:  # floor, ghost
            contype, conaffinity = 0, 0
            if body in ghost_bodies:
                model.geom_rgba[g] = (0.15, 0.85, 0.25, 0.45)
        model.geom_contype[g], model.geom_conaffinity[g] = contype, conaffinity
        model.geom_friction[g] = (friction, torsional, 0.0)
        if contype or conaffinity:
            model.geom_condim[g] = physics.condim
            model.geom_solref[g] = physics.solref
            model.geom_solimp[g] = physics.solimp
            model.geom_margin[g], model.geom_gap[g] = 0.0, 0.0

    ids = lambda names, prefix="": np.array([model.joint(prefix + n).qposadr[0] for n in names])  # noqa: E731
    data = mujoco.MjData(model)
    return Scene(
        model=model,
        data=data,
        qpos_adr=ids(JOINTS),
        qvel_adr=np.array([model.joint(n).dofadr[0] for n in JOINTS]),
        actuators=np.array([model.actuator(f"{n}_pos").id for n in JOINTS]),
        ghost_qpos_adr=ids(JOINTS, "ghost_") if with_ghost else None,
        hand_bodies=hand_bodies,
        arm_bodies=arm_bodies,
        table_geom=table_geom,
        excluded_hand_pairs=excluded,
        physics=physics,
        tube_bodies=tube_bodies,
    )


def _find_design_overlaps(scene: Scene, rest_poses) -> set[frozenset[str]]:
    """Hand body pairs in contact in the zero pose or the given rest poses, other than
    pairs on two different fingers (those are real collisions and stay on)."""
    pairs = set()
    for q in (np.zeros(len(JOINTS)), *rest_poses):
        set_joints(scene, q)
        for c in scene.data.contact[: scene.data.ncon]:
            b1, b2 = int(scene.model.geom_bodyid[c.geom1]), int(scene.model.geom_bodyid[c.geom2])
            if b1 in scene.hand_bodies and b2 in scene.hand_bodies:
                n1, n2 = scene.body_name(b1), scene.body_name(b2)
                f1, f2 = _finger(n1), _finger(n2)
                if f1 is None or f2 is None or f1 == f2:
                    pairs.add(frozenset((n1, n2)))
    return pairs


def set_joints(scene: Scene, q: np.ndarray, ghost: bool = True) -> None:
    """Put the robot (and the ghost) at ``q`` at rest, with matching drive targets."""
    scene.data.qpos[scene.qpos_adr] = q
    scene.data.qvel[:] = 0.0
    scene.data.ctrl[scene.actuators] = q
    if ghost and scene.ghost_qpos_adr is not None:
        scene.data.qpos[scene.ghost_qpos_adr] = q
    mujoco.mj_forward(scene.model, scene.data)
