# Frozen codec-search AR: 1,000-video confirmation (research artifact)

This package preserves the **frozen, task-specific** H.264/H.265 experiment.
It is suitable for reproducing the reported `r2plus1d_18` Top-1 BD-rate, **not**
for claiming a model-agnostic or low-latency production video codec. There is
no newly trained `.pth` checkpoint: the frozen checkpoint is the policy
configuration in [`configs/codec_search_ar_frozen_v1.json`](../../configs/codec_search_ar_frozen_v1.json),
the pinned code/torchvision weights, and the calibration thresholds.

| Codec | Matched `r2plus1d_18` Top-1 BD-rate | Paired video bootstrap 95% CI | Cross `r3d_18` Top-1 BD-rate |
|---|---:|---:|---:|
| H.264 | **−24.88%** | **[−26.82%, −22.94%]** | −1.03% |
| H.265 | **−16.09%** | [−17.44%, −14.74%] | −0.99% |

Both point estimates are below the predeclared −15% engineering threshold on
the matched analyzer. Only H.264's bootstrap interval is wholly below −15%.
On `r3d_18`, both intervals include zero and Top-1 drops at some QPs; transfer
to another AR analyzer is **not established**.

## What was frozen and measured

- Pilot code: `e8f29d6300df576a8c8869b6befcf9fd9d957548`; 1,000-video runner:
  `0fab4b95ae65e2e93ff1985a6491e2e3bf38dec1`.
- Public dataset: `qktttttttttt/kineticscleaned`; canonical hash-split TEST,
  deterministic class-balanced 1,000-video fingerprint `aae3888f3ae34d08`.
  TEST was held out from *this pilot's* 400-TRAIN calibration and 104-VAL
  screen. Earlier project-wide work may have inspected this TEST partition.
- Same 1,000 videos for both codecs; five QPs `30,35,40,45,50`; 16 frames at
  128×128; FFmpeg `libx264`/`libx265` preset `medium`; frozen torchvision
  `R2Plus1D_18_Weights.KINETICS400_V1` and `R3D_18_Weights.KINETICS400_V1`.
- Encoder tries six fixed pixel candidates at each QP and chooses using actual
  encoded bytes plus label-free `r2plus1d_18` constraints. `r3d_18` is never
  consulted by the selector. Every bpp uses the original 128×128 denominator.
- Four disjoint 250-video shards are computational only. The final curves are
  formed from all **1,000 unique videos**, then BD-rate is computed once. The
  2,000 bootstrap draws resample complete videos, not individual QP points.

## Package contents and verification

- `manifest.json`: commit/fingerprint provenance, Kaggle notebook URLs,
  measured results and SHA-256 checksums.
- `h264_result.json`, `h265_result.json`: aggregate curves, counts, metrics and
  paired-video bootstrap intervals.
- `h264_shards.tar.gz`, `h265_shards.tar.gz`: allowlisted per-video records,
  shard reports, frozen policy copies and run logs. No videos, model weights or
  API credentials are bundled. Each archive extracts as `shard_0/` ...
  `shard_3/`, each containing `shard_records.jsonl` and `shard_result.json`.

To recalculate a codec result after extraction, run from the repo root (replace
`h264` with `h265` as needed):

```bash
mkdir -p /tmp/codec-confirm-h264
tar -xzf results/codec_search_ar_confirm_1000/h264_shards.tar.gz -C /tmp/codec-confirm-h264
python ops/merge_codec_search_confirm.py \
  --shard-dir /tmp/codec-confirm-h264/shard_0 \
  --shard-dir /tmp/codec-confirm-h264/shard_1 \
  --shard-dir /tmp/codec-confirm-h264/shard_2 \
  --shard-dir /tmp/codec-confirm-h264/shard_3 \
  --out /tmp/codec-confirm-h264/recomputed.json --bootstrap 2000
```

The [preregistered protocol](../../docs/RUN_DESIGN_CODEC_SEARCH_CONFIRM_1000.md)
defines the primary threshold. This implementation spends compute on up to six
real encodes and task-analyzer passes per clip/QP; runtime and energy were not
benchmarked, so BD-rate gains alone do not establish deployment readiness.
