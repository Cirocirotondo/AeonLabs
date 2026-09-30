# Intrinsics are estimated from the reference paper once, then fixed

We have no calibration of the camera: the phone model, lens, video mode and the step that turned the iPhone video into 832×480 are all unknown, and the hand pose estimator does not export the intrinsics it used. So the focal length is estimated from the shape of the reference paper, pooled over every clip in the first `--glob` run, starting from an iPhone 1x main camera prior (about 700 px for an 832-wide frame, allowed 580–890 px). The result is saved and reused for every later human demonstration, on the assumption that all of them come from the same phone with the same settings. We also assume the principal point is at the image centre and there is no lens distortion, and that the frame is a 16:9 original stretched to its size without cropping, so fy = (16/9)·(height/width)·fx (1.026 for 832×480). We first assumed square pixels; on the first clip the stretch lowered the corner residual from 1.28 to 0.90 px, and with square pixels the sheet's best-fitting aspect ratio came out 1.45 instead of A4's 1.414, which is the same 2.6% stretch.

## Considered Options

- **Checkerboard calibration of the phone**: rejected for now because the phone and its settings are unknown, and the delivered video may be cropped or resized after capture.
- **Intrinsics from the hand pose estimator**: this would put the skeleton and the camera pose in the same camera model, but the estimator's intrinsics are not available.
- **A new focal length per clip**: rejected because top-down views of a single rectangle constrain the focal length poorly, while pooling tilted views over many clips constrains it well.

## Consequences

On the first clip the paper barely constrains the focal length (the corner residual changes by under 1% between 450 and 580 px), so the estimate sits on the 580 px lower bound. The hand pose estimator's CSV independently implies about 585 px: its 3D wrist projects onto the overlay's wrist with that focal length.


Every saved camera pose depends on this focal length. If a clip comes from another phone, lens, zoom or video mode, or if the resize turns out to be a stretch (fy ≈ 1.026·fx), its poses will be biased and nothing will flag it. Re-run with `--estimate-focal` (and `--pixel-aspect` if the video was not a stretched 16:9 original). Replace this ADR when real calibration or the estimator's intrinsics become available.
