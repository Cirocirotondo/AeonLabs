# Context map

- [paper_extrinsics/](paper_extrinsics/CONTEXT.md): camera pose per video frame from a reference paper on the table.
- [hand_retarget/](hand_retarget/CONTEXT.md): human hand skeletons from video turned into UR5e + DG5F joint angles (robot demonstrations), with a kinematic replay viewer.
- [physics_sim/](physics_sim/CONTEXT.md): robot demonstrations replayed on the simulated robot with physics, before the real robot.
- [real_robot/](real_robot/README.md): robot demonstrations replayed on the real UR5e + DG5F through the ur5 PC's existing controllers.

hand_retarget's robot demonstration export is the input of physics_sim and then of real_robot. paper_extrinsics will later give hand_retarget a measured table and camera.
