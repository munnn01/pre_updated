# CRC-V6 rescue: checkpoint crossover and structural DCT projection

## Decision question

CRC-V5 moved every codec/QP dual but left real-codec overhead at +21--33%.
This round asks whether either of two *new mechanisms* can meet the relaxed AR
continuation rule without reading the test split:

1. **Crossover:** canonical `dieulinhh` Stage-1 bytes followed by the `t0_lr5`
   calibration that was directionally strongest for H.265.
2. **DCTP:** constrain deployment output to attenuation of weak high-frequency
   DCT coefficients, optionally retaining 25% of the semantic residual and/or
   stabilizing source-static pixels over time.

Negative Top-1 BD-rate is better. A candidate passes only when **both H.264 and
H.265 are negative and at least one codec is <= -15%**. It must additionally
lose no more than 0.05 absolute Top-1 at any QP against its raw-codec anchor.

## Phase A: requested crossover on all five mandatory accounts

All five accounts receive the exact same immutable `dieulinhh` Stage-1 file
(SHA-256 `a2ad614b874d756db4b5db5fb2a0cefbae300353ed3044a3c27652d0f1b0e5f2`).
Within each account, both branches start from those same bytes:

- control: fixed `beta=0.001` STE calibration;
- treatment: target added-rate 0%, `dual_lr=0.005` (`t0_lr5`).

Accounts: `shungg05`, `dieulinhh`, `huolgggnuyen`, `baooo25r`,
`trmnguyn111`. The evaluation sample is a new deterministic validation shard,
salt `crc-v6-crossover-v1`, shard `0/10`, held-out R2Plus1D-18, real x264/x265
medium, QP `[30,35,40,45,50]`. The already inspected CRC-V5 shard is not reused.

This is an adaptive rescue hypothesis because its two components were chosen
after CRC-V5. A pass is therefore a validation result, not a final test claim.

## Phase B: DCTP 2^3 mechanism screen

The screen uses a full factorial with factors:

| Factor | Low | High |
|---|---:|---:|
| retained semantic residual | 0.00 | 0.25 |
| weak-HF DCT attenuation | 0.60 | 1.00 |
| static-region temporal stabilization | 0.00 | 0.35 |

Fixed values: threshold multiplier 1.0, block 8, high band `u+v>=4`, QP slope
0.65, H.264/H.265 scale 1.0, motion tau 0.05. Every arm uses the same canonical
checkpoint bytes. Run order was randomized with seed `220921`; `pyDOE3` was not
available in the local environment, so the exact full factorial was generated
with `itertools.product` and shuffled with Python's seeded `random.Random`.

| Account | residual | DCT | temporal |
|---|---:|---:|---:|
| `shungg05` | 0.25 | 1.00 | 0.00 |
| `dieulinhh` | 0.25 | 0.60 | 0.00 |
| `huolgggnuyen` | 0.00 | 0.60 | 0.35 |
| `baooo25r` | 0.00 | 1.00 | 0.00 |
| `trmnguyn111` | 0.00 | 0.60 | 0.00 |
| `wagur124705` | 0.25 | 0.60 | 0.35 |
| `htran123456` | 0.00 | 1.00 | 0.35 |
| `vtk269` | 0.25 | 1.00 | 0.35 |

The five mandatory accounts are therefore present in both the crossover and
the structural screen. Account is a run/block nuisance, not a clip-level
replicate; per-clip rows are repeated measurements within a run.

## Sequential rule

1. Run Phase A and Phase B on disjoint salted validation screens only.
2. Reject any candidate that misses either codec sign, the one-codec `-15%`
   threshold, or the per-QP Top-1 guardrail.
3. If one or more pass, freeze the winning configuration using the prespecified
   ordering: largest worst-codec saving, then smallest worst-QP Top-1 loss.
4. Replicate the frozen winner on disjoint validation clips and at least three
   checkpoint seeds before reading the 1,159-clip test split.
5. If neither mechanism passes, do not tune more dual learning rates on the same
   shard; close additive/dual rescue and move to an independently trained
   subtractive preprocessor.
