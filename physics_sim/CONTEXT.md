# Physics simulation

Replays a robot demonstration on a simulated UR5e + DG5F with physics, to catch what would go wrong on the real robot.

## Language

**Robot demonstration**:
A time series of robot joint angles (arm and hand) at 60 Hz, in simtoolreal_newton's layout; here, the commands the robot is asked to follow.
_Avoid_: Trajectory (bare), motion file

**Physics replay**:
Sending a robot demonstration's joint angles to the simulated robot's position drives and letting physics (drive gains, effort limits, gravity, contacts) decide where it goes.
_Avoid_: Replay (bare, in hand_retarget that is the kinematic replay), rollout (that is a policy acting), sim2sim (in simtoolreal that is replaying a policy in another engine)

**Ghost**:
A second, contact-free copy of the robot beside the real one, posed exactly at the commanded joint angles.
_Avoid_: Reference robot, target robot

**Tracking error**:
The difference between a joint's commanded and actual angle during a physics replay.
_Avoid_: Fingertip error (that is hand_retarget's retargeting measure), lag

**Design overlap**:
Two links of the same hand whose collision meshes touch at rest, around the joint between them; excluded from contact, since the real parts do not collide there.
_Avoid_: Self-collision (that is a real collision between parts of the robot)
