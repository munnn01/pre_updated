# Frozen codec-search AR confirmation on 1,000 clips

This is a confirmation of `codec_search_ar_v1`, not another tuning run. The
frozen manifest is `configs/codec_search_ar_frozen_v1.json`. The pilot used
400 TRAIN clips to calibrate each codec and 104 VAL clips to screen them.
There is no newly trained `.pth`: the checkpoint consists of the original
code commit, the frozen torchvision weights and the two fixed tolerances.

## Primary question and rule

For each codec, compare the label-free policy with `identity128` on the same
1,000 TEST source videos at QP 30, 35, 40, 45 and 50. The primary endpoint is
Top-1 BD-rate on the matched frozen `r2plus1d_18` analyzer, computed from
aggregate bpp and Top-1 curves. The requested engineering threshold is met
if **either H.264 or H.265 is strictly below -15%**. The other codec need not
be negative. Report a paired video bootstrap 95% interval separately; do not
silently reinterpret the threshold as a confidence-bound criterion.

The 1,000 videos are selected deterministically and approximately class
balanced from the canonical hash-split TEST partition, salt
`codec-search-confirm-test-v1`. This partition was not used in this pilot's
calibration or 104-clip screen. Prior project-wide exploration may have
inspected TEST; call this a held-out confirmation **relative to this pilot**,
not a globally untouched benchmark. Each video is one bootstrap unit, with
its five QPs and both arms resampled together. Do not treat 5,000 QP points
as independent videos.

## Frozen procedure

- The source is 16 frames at 128x128 with deterministic centered temporal
  sampling and stride 2. FFmpeg uses `libx264`/`libx265`, preset `medium`.
- At each QP the encoder evaluates the same six candidates as the pilot,
  measures elementary-stream bytes, and chooses using the fixed tolerances
  from the manifest. No TEST labels enter candidate selection.
- All bpp use the original 128x128 denominator, including lower-resolution
  streams. The cross-backbone `r3d_18` sees the identity and selected decode
  only; it does not influence selection.
- Run H.264 first as four disjoint 250-video shards. If its aggregate result
  passes the predeclared threshold, H.265 is optional for the stated goal.
  Otherwise run H.265 with the same frozen sampling and shard scheme.
- Sharding is computational only. Never average four shard BD-rates; merge
  per-video records and recompute one five-QP curve over all 1,000 videos.
- Archive the manifest hash, selected video fingerprint, code commit, codec,
  QPs, curves, candidate counts, per-video records and complete run logs.

Interpret a negative point estimate on 1,000 clips as satisfying the user's
engineering criterion, but distinguish it from robust cross-model transfer,
which failed the 104-clip pilot and remains a secondary endpoint.
