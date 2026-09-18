# OD/AR generalization matrix after the QP45 result

Status: frozen before launch.  Date: 2026-09-18.

The selected OD method is background Gaussian sigma 4 with an 8-pixel context
halo and decoder Gaussian POST sigma 1 only at QP >= 45.  On the first held-out
500-image split it reached -13.145% H.264 and -8.017% H.265 BD-rate.  The next
stage tests whether that result survives a new sample and different analyzers;
it does not reopen sigma, halo, QP, score, or dilation tuning.

| Kaggle account | Frozen question | Run |
|---|---|---|
| `wagur124705` | New-sample H.264 replication | n=500, seed 20260919, H.264 |
| `htran123456` | New-sample H.265 replication | n=500, seed 20260919, H.265 |
| `shungg05` | Transfer to FCOS evaluator | n=200, seed 20260918, both codecs |
| `vtk269` | Transfer to RetinaNet evaluator | n=200, seed 20260918, both codecs |
| `hoangminhhuy123` | Dependence on encoder-side mask analyzer | ResNet50 mask, MobileNet evaluator, n=200 |
| `nguyenhoanglan1232` | Transfer of QP45 decoder denoising to action recognition | r3d_18, n=200 Kinetics clips |

All OD runs retain the same five QPs, real x264/x265 medium preset, score 0.20,
dilation 0.30, feather 8, halo 8, and exact-identity ROI.  FCOS/RetinaNet and
the reversed Faster R-CNN pair remain held out from the mask analyzer.

The AR run is deliberately POST-only: it shares the exact decoded anchor and
bpp, then applies Gaussian sigma 1 at QP45/50.  This isolates transfer of the
denoising mechanism without repeating the rejected detector-tube encoder.
Target-class probability is the primary continuous AR metric; top-1 is
secondary because n=200 is coarse.

## Reading rules

1. Replication passes only if BD-rate remains negative and the OD gap stays at
   or above -0.05 for its codec.
2. Cross-evaluator arms are mechanism screens.  A non-negative codec or gap
   failure closes that evaluator; no retuning on the same sample.
3. AR advances only if target-probability BD-rate is negative on both codecs
   and top-1 does not materially regress at any enabled QP.
4. Every run persists per-image or per-clip records.  Point estimates are not
   promoted to a final claim without paired uncertainty analysis.

