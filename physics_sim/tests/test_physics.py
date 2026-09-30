"""The simulated robot holds still poses, follows slow motions, and the report finds contacts."""

import numpy as np
import pytest

from physics_sim.replay import RobotDemonstration, replay, report
from physics_sim.scene import JOINTS, build_scene, set_joints

HOME = np.array([-1.5708, -1.05, 1.95, -0.9, 1.571, -2.618] + [0.0] * 20)


def demonstration(q):
    return RobotDemonstration("test", np.arange(len(q)) / 60.0, np.asarray(q), 0.515)


@pytest.fixture(scope="module")
def scene():
    return build_scene(with_ghost=False, rest_poses=(HOME,))


def test_nothing_touches_at_home(scene):
    set_joints(scene, HOME)
    assert scene.data.ncon == 0


def test_a_still_pose_is_held(scene):
    log = replay(scene, demonstration([HOME] * 60))
    assert np.degrees(np.abs(np.array(log.q) - HOME)).max() < 0.5
    assert not log.contacts


def test_a_slow_finger_motion_is_followed(scene):
    q = np.repeat(HOME[None], 120, axis=0)
    q[:, JOINTS.index("rj_dg_3_3")] = np.linspace(0.0, 1.0, 120)  # middle PIP, 1 rad in 2 s
    log = replay(scene, demonstration(q))
    assert np.degrees(np.abs(np.array(log.q) - np.array(log.target))).max() < 5.0


def test_the_report_names_a_finger_collision(scene):
    # Index and middle spread toward each other: their proximal phalanges collide.
    q = HOME.copy()
    q[JOINTS.index("rj_dg_2_1")] = 0.6
    q[JOINTS.index("rj_dg_3_1")] = -0.6
    q[JOINTS.index("rj_dg_2_2")] = q[JOINTS.index("rj_dg_3_2")] = 0.3
    log = replay(scene, demonstration([q] * 30))
    assert "finger-finger" in log.contacts
    assert "rl_dg_2" in report(log) and "rl_dg_3" in report(log)
