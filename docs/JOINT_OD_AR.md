# Joint Object Detection + Action Recognition direction

## Implemented foundation

`src/models/importance_tube.py` supplies one parameter-free transform for both
regimes:

```text
source frames -> frozen detector -> per-frame masks -> short object tubes
              -> feathered spatial suppression outside objects
              -> motion-gated stabilization of static background
              -> frozen x264/x265
```

- `T=1`: exactly the OD R0 transform when tube/feather/temporal options are zero.
- `T>1`: protects people and manipulated objects across short detection misses;
  only static background is pulled toward the previous processed frame.
- The original detector mask is reapplied after every operation, so its protected
  core remains pixel-exact.

`ops/probe_action_tubes.py` is the first decision gate. It applies this mechanism
to Kinetics without training and reports BD-rate on both top-1 and target-class
probability. A learned joint model should not be started until this probe shows
that object tubes improve or preserve AR while reducing rate.

## Measurement contract

1. The network producing encoder-side masks must differ from the evaluation
   analyzer. OD R0 defaults to MobileNetV3-FPN masks and ResNet50-FPN evaluation.
2. Select sigma/dilation/tube radius on validation data, freeze them, then report
   once on test data.
3. CI is a paired image bootstrap with replacement. Repeated images receive new
   COCO ids so multiplicity is not silently deduplicated.
4. Report every QP gap as well as BD-rate. A negative BD value does not override
   an accuracy collapse at one operating point.
5. Use both x264 and x265; a single-codec win is diagnostic, not a universal claim.

## Next training rung (not claimed by this repository yet)

Alternate COCO `T=1` batches and Kinetics `T=16` batches through a shared
importance/suppression trunk. Keep small task adapters and codec-specific POST
heads; the existing AR evidence shows that a shared POST trunk destroys useful
codec specialization. The target objective is:

```text
L = lambda_det * L_detection + lambda_ar * L_action + beta * rate
  + tau * L_background_temporal + eta * L_mask_area
```

Treat the task metrics as constraints: neither mAP nor top-1 may fall more than
0.05 at any QP. The learned gate may scale suppression only outside the protected
core; it must not synthesize or reshape object pixels.
