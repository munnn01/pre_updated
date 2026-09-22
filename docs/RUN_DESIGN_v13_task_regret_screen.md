# V13: QP-conditioned task-regret mechanism screen

## Question

V12 moved mean H.264 BD-rate from +1.916% to -0.077% but pivoted the R-D
curve, while H.265 saved bits by sacrificing 2.9--7.4 Top-1 percentage points.
V13 screens one algorithmic treatment: reverse the constrained problem from "minimize task loss
subject to rate no worse than anchor" to "minimize rate subject to task loss no
worse than anchor."

For codec `c` and QP `q`, the raw-codec anchor is the unedited clip encoded by
the same real codec/QP.  The treatment solves

```text
minimize   R(pre+codec) / R(raw+codec) - 1
subject to CE(pre+codec) - CE(raw+codec) <= epsilon[c,q]
```

with one projected dual `mu[c,q]` per cell.  The registered screen uses
`epsilon=0`, `dual_lr=0.001`, `mu_init=0.1`, `mu_max=10`, and
`rate_weight=1`.  All V12 architecture, DCT, distortion, optimizer LR, QP
schedule and codec settings are held fixed.

## Training and screen

- Start from the audited 500-step last checkpoint.
- Reset optimizer and old rate duals because their moments/state belong to the
  old objective. This makes the first screen exploratory rather than a perfectly
  paired causal comparison with V12, which resumed those states.
- Train three complete epochs (12,948 steps) with a fresh task-dual schedule.
- Train H.264 and H.265 separately.
- Evaluate the final checkpoint, not a validation-selected epoch.
- Screen on shard 0/5 of the locked 1,010-clip validation set using salt
  `v13-task-regret-screen-v1`: exactly 208 clips, fingerprint
  `f321b810af2c507c`.
- Use held-out `r2plus1d_18` and real codec QPs 30/35/40/45/50.

Initial allocation is deliberately only two jobs:

| Account | Codec | Seed | Source |
|---|---|---:|---|
| `qktttttttttt` | H.264 | 292001 | private seed-282001 500-step checkpoint |
| `baoancut` | H.265 | 292002 | frozen public H.265 500-step checkpoint |

`trnhlng` is reserved for replication and is not used during the mechanism
screen.

## Registered decision rule

Advance a codec only when all conditions hold on the 208-clip screen:

1. BD-rate is at most -5%;
2. BD-accuracy is non-negative; and
3. no evaluated QP loses more than 0.5 percentage point Top-1 against its
   same-codec anchor.

Failing any condition is a futility stop for that codec. Passing is only
exploratory evidence: the next stage must run a fresh-optimizer control and
treatment from the same checkpoint, replicate on independent seeds over all
1,010 validation clips, then make one final untouched evaluation on the
1,159-clip test split.
No threshold or hyperparameter may be changed after looking at screen output.
