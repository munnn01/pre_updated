# One-cell Kaggle run: Object Detection + Action Recognition

Attach these two inputs to a Kaggle notebook and enable a T4 GPU + Internet:

- `awsaf49/coco-2017-dataset`
- `qktttttttttt/kineticscleaned`

Create one code cell, put `%%bash` on its first line, then paste the complete
contents of [`kaggle/joint_probe_cell.sh`](../kaggle/joint_probe_cell.sh).
The checked-in template contains `REF="__REF__"`; replace `__REF__` with `main`
for an interactive run or with a commit SHA for a reproducible run.

The default `quick` profile uses 50 COCO images and 20 Kinetics clips.  Change
the first assignment to `PROFILE="confirmatory"` for 500 images, 200 clips, the
full five-QP grid, and 1,000 paired bootstrap draws.  Both profiles evaluate:

1. held-out OD background suppression (MobileNet-FPN creates masks,
   ResNet50-FPN measures COCO mAP), and
2. OD-derived spatio-temporal importance tubes with a frozen Kinetics analyzer.

Each stage writes a dedicated log, and the exit trap creates the final artifact
even when a probe fails. The downloadable artifact is
`/kaggle/working/joint_probe_outputs.tgz`; uncompressed JSON diagnostics remain
under `/kaggle/working/outputs/` even if notebook stdout is truncated.

To generate and push the same one-cell notebook through the Kaggle API:

```bash
python ops/push_joint_probe.py --commit <git-sha> --account <kaggle-user>
```
