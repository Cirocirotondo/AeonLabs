# Camera pose from the reference paper

Finds where the handheld camera is, in every frame of a human demonstration video, from an A4 sheet lying still on the table.

## Language

**Reference paper**:
The A4 sheet (210 × 297 mm) lying still on the table throughout a human demonstration, used as the known object that locates the camera.
_Avoid_: Marker, target, sheet (bare)

**Paper frame**:
The world frame of a human demonstration: origin at the centre of the reference paper, z up out of the table, x along the long edge pointing away from the person in the first frame.
_Avoid_: World frame (bare), table frame, origin

**Camera pose**:
Where the camera is in the paper frame at one video frame (the transform taking camera-frame points into the paper frame). Changes every frame because the camera is handheld.
_Avoid_: Extrinsics (in prose), camera transform, view

**Intrinsics**:
The camera's focal length and principal point, which turn a camera-frame point into a pixel. A property of the camera shared by every human demonstration: found once and reused, not found again per video.
_Avoid_: Calibration (bare), K (in prose)

**Pose source**:
How a frame's camera pose was obtained: from the reference paper seen in that frame, carried over from neighbouring frames by tracking the table, or failed.
_Avoid_: Status, quality flag
