# CRC-V5: codec-native constrained AR screening

## Status and question

Pre-registered before launch. QPC-V4 learned its QP condition and increased
Top-1 at every QP, but all real-codec BD-rates stayed positive because added
bitrate remained excessive. CRC-V5 asks one narrow question: **does optimizing
the measured added-rate constraint through a real-codec STE move validation
BD-rate below the QPC warm start?**

This is a screening experiment. Its ~50-clip validation shard and single
warm-start seed cannot support a final performance claim.

## Experimental unit and blocking

- Independent treatment unit: one complete fine-tune job.
- Common starting state: QPC-V4 uniform seed 1, best epoch 14.
- Common data/split/evaluator: Kinetics hash fingerprint `30f083f8520a`, salted
  validation shard `0/20`, held-out R2Plus1D-18.
- Within each job, training uses randomized complete blocks containing every
  `2 codecs x 5 QPs` cell exactly once. This prevents codec/QP from being
  confounded with optimizer time.
- The five jobs share seed, batch size, 500 calibration steps, LR, architecture
  and evaluator. Only the registered rate-constraint arm differs.

## Arms and assigned accounts

| Account | Arm | Target added-rate | Dual LR |
|---|---|---:|---:|
| `shungg05` | STE control | none; fixed beta=0.001 | none |
| `dieulinhh` | `t0_lr1` | 0% | 0.001 |
| `huolgggnuyen` | `tm5_lr1` | -5% | 0.001 |
| `baooo25r` | `t0_lr5` | 0% | 0.005 |
| `trmnguyn111` | `tm5_lr5` | -5% | 0.005 |

The four constrained jobs form a 2x2 factorial. The control estimates the
effect of real-codec calibration and codec conditioning without adaptive duals.

## Mechanism

At each step the sampled raw clip and processed clip are encoded by the same
real codec and QP. For cell `c,q`:

`g = bpp(processed) / bpp(raw) - 1 - target`

`lambda[c,q] <- clip(lambda[c,q] + dual_lr * g, 0, 1)`

The loss contains `lambda[c,q] * g`. Its forward value uses real x264/x265
bits; its backward path uses the existing differentiable codec proxy via STE.
FiLM receives `[QP, is_h264, is_h265]`, with codec columns zero-initialized so
the warm start is initially identical for both codecs.

## Registered screening gates

Negative BD-rate is better.

1. Infrastructure validity: all jobs use the same commit, warm-start SHA-256,
   validation IDs, anchor curves, codec preset and evaluator.
2. Constraint mechanism: final duals are finite, at least one dual moves from
   `lambda_init`, and measured added-rate is reported for every codec/QP cell.
3. Directional continuation: a constrained arm must beat the STE control by at
   least 2 BD-rate percentage points on **both** H.264 and H.265, without losing
   more than 0.05 Top-1 at any QP.
4. Absolute continuation: both codec BD-rates must be negative on the screen.
5. The desired `<= -15%` target is not relaxed. If a screen passes gates 1-4,
   repeat the winning arm with three independent seeds on the full validation
   split before touching the 1,159-clip test set.

Failure of all four constraint arms closes this exact scalar-dual design. It
does not justify increasing epochs or reading the test split.

