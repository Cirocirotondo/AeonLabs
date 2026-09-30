# physics_sim

Replays a robot demonstration (hand_retarget's export) on the UR5e + DG5F in MuJoCo **with physics**, to see what
the real robot would do before it does it. Vocabulary is in [CONTEXT.md](CONTEXT.md).

```bash
# viewer: the robot follows the commands, the green ghost beside it shows the commanded pose
uv run physics-sim-replay ../hand_retarget/out/20260928T132405_scene0_vector.npz
uv run physics-sim-replay <demo.npz> --loop        # replay again until the window is closed
uv run physics-sim-replay <demo.npz> --speed 0.25  # a quarter of real time (the physics is unchanged)
uv run physics-sim-replay <demo.npz> --test        # table and tube 2 cm higher, the hand passes through the table
uv run physics-sim-replay <demo.npz> --headless    # no window: replay once and print the report
uv run physics-sim-replay <demo.npz> --headless --save-plots out/   # ... and save the joint figures as PNG
```

The joint targets go to position drives at 60 Hz, physics at 8 substeps. Drives, gains, gravity compensation,
friction and contact parameters are those of simtoolreal_newton's MuJoCo sim2sim, and the robot base and table
are placed as there (the table height comes from the demonstration file). Unlike simtoolreal's training, hand-table,
arm-table and finger-finger contacts are on: the point is to find what the real robot would hit. Hand links that
already overlap at rest (neighbouring meshes at a joint) are excluded.

After each replay it prints a report: joint tracking error, drives at their effort limit, and every contact
(hand-table, arm-table, finger-finger, hand-self) with how often, how deep and how hard. With the viewer it also
opens two figures of actual vs target joint angles, one for the arm and one for the hand (`--no-plots` to skip).

The robot's URDF and meshes are read from the sibling `hand_retarget/assets`.

Run the tests with `env -u PYTHONPATH uv run pytest`.
