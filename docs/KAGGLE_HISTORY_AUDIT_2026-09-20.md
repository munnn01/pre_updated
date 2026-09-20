# Kaggle history audit and next experiment decision (2026-09-20)

## Decision

Run **QPC-V4 as a matched 3-family x 3-seed experiment for 16 epochs**. Do not
return to ROI/tube heuristics. Keep OD as the frozen positive baseline and use
AR QPC as a mechanistic experiment, not as a promise that -15% BD-rate will be
reached.

The three families are:

1. unconditioned additive retraining control;
2. additive + scalar QP FiLM with uniform QP sampling;
3. the same QP FiLM with preregistered BD-weighted QP sampling.

Only architecture/conditioning and sampling weights may differ. Dataset split,
teachers, loss, virtual codec, optimizer, epochs and seeds remain matched.

## Audit scope and limits

- Accounts audited: 14 (`dieulinhh`, `dngbolm`, `hieusunday0412`,
  `hoangminhhuy123`, `htran123456`, `huolgggnuyen`, `linhowi05`,
  `nguyenhoanglan1232`, `ngynanhthuw`, `shungg05`, `thuha1205`,
  `tranthihongdieu`, `vtk269`, `wagur124705`).
- Visible notebooks: 401; declared notebook versions: 723.
- Current sources retrieved: 401/401. There were 123 multi-version notebooks.
- Historical private sources not retrievable: 322 version bodies; Kaggle returned
  HTTP 403 for every historical-version request. Current source and the declared
  version count were still verified. Locally cached old versions were used where
  present, but this is not a claim that all 723 source bodies were readable.
- Project-relevant output endpoints: 366; downloaded: 358; unavailable: 8
  obsolete endpoints returning HTTP 404; successful endpoints with no matching
  compact result/log file: 3.
- Download manifest recorded 1,593 artifact paths. The local result tree contains
  911 JSON and 44 NPZ files; all 955 parsed successfully. Many outputs are
  duplicates because old notebooks copied a repository/output tree into
  `/kaggle/working`.
- Source search found **no executed QPC training command** using
  `additive_qpc.yaml` or `model.arch=additive_cond`. Existing mentions are
  configs, model/evaluator support and preregistration text. Therefore QPC has
  not been tested yet.

The eight unavailable output endpoints are recorded by
`ops/download_kaggle_results.py`; this limitation does not affect the identified
OD, AR-V1/V2, ROI-V3 or learned-additive result families used below.

## Evidence consolidated from raw outputs

### OD: retain as a frozen baseline

| Evaluation | N | H.264 BD-rate | H.265 BD-rate | Worst mAP gap |
|---|---:|---:|---:|---:|
| original held-out | 500 | -13.15% | -8.02% | -0.0065 |
| new sample | 500 | -15.41% | -7.96% | -0.0061 |
| FCOS cross-evaluator | 200 | -7.12% | -7.30% | -0.0206 |
| RetinaNet cross-evaluator | 200 | -7.29% | -5.12% | -0.0145 |
| reversed analyzer/evaluator | 200 | -8.24% | -3.15% | -0.0140 |

OD has repeatably negative point estimates and passes the accuracy guardrail,
but it does not support a general “better than -15%” claim, especially on H.265
or cross-evaluator tests. Further grid searching was correctly stopped to avoid
selection bias. The auditable claim is robustness with moderate savings, not a
universal -15% improvement.

### AR heuristics: close these branches

- Decoder Gaussian transfer: H.264 `+5.24%`, H.265 `+2.88%` BD-rate; harmful.
- Initial action tubes: H.264 `+9.80%`, H.265 `+8.57%`; harmful.
- Guarded AR-V2 best safe points ranged from about `-0.27%` to `-7.14%` on
  H.264 and `-0.06%` to `-1.69%` on H.265. No arm reached -15%; paired
  bootstrap probability of reaching -15% was effectively zero (maximum 0.008).
- ROI-V3 used 200 evaluation clips/job and has no training loop. Its failure
  cannot be repaired by “training longer.” Across six codec/backbone jobs, the
  best ROI-arm point estimates were approximately `-6.29`, `-4.90`, `-3.96`,
  `-3.10`, `-2.05`, and `-1.08` percent. Every local gate failed.

### Learned additive AR: the realistic reference range

Independent/raw result variants put the strongest learned AR methods roughly in
the `-4%` to `-7%` H.264 and `-2%` to `-4%` H.265 range. Two useful corrected
references are:

- v9b full: H.264 `-5.89%` (95% CI `[-8.54,-3.09]`), H.265 `-3.56%`
  (`[-5.39,-1.62]`);
- E4 confirmatory (N=11,724): H.264 `-4.61%`, H.265 `-3.89%`.

This history makes a direct jump to -15% from scalar QP conditioning unlikely.
QPC is still justified because the old kappa10 audit found QP30 had roughly
4.1x--10.2x worse gap-per-added-bit efficiency than the best QP: the existing
single editor is measurably mismatched across operating points.

## Dataset and training diagnosis

The canonical cleaned split is 8,632 train / 1,010 validation / 1,159 test
clips over 400 classes (about 21.6 train clips/class). It is much smaller than a
full Kinetics training set, so data scale can limit generalization. It does **not**
explain ROI-V3, which is a non-learned evaluation. It also should not be changed
inside the first QPC comparison, because changing data size and architecture at
the same time would make the causal result uninterpretable.

The valid additive-kappa10 lineage trained for 16 epochs, about 1,079 optimizer
steps/epoch or 17,264 steps total. Therefore the earlier six-epoch QPC draft was
underpowered. QPC-V4 now uses the same 16-epoch budget and patience 4.

## Frozen QPC-V4 matrix

| Account | Family | Seed |
|---|---|---:|
| `wagur124705` | additive control | 0 |
| `htran123456` | additive control | 1 |
| `dngbolm` | additive control | 2 |
| `hoangminhhuy123` | QPC uniform | 0 |
| `shungg05` | QPC uniform | 1 |
| `hieusunday0412` | QPC uniform | 2 |
| `vtk269` | QPC BD-weighted | 0 |
| `nguyenhoanglan1232` | QPC BD-weighted | 1 |
| `linhowi05` | QPC BD-weighted | 2 |

Uniform samples QP `[30,35,40,45,50]` equally. BD-weighted uses point weights
`[0.089,0.215,0.288,0.285,0.123]`, derived from the preregistered four segment
weights `[0.178,0.252,0.324,0.246]`. Validation evaluates all QPs equally, so
training sampling cannot hide a weak operating point.

## Gates and falsification

Training uses train/validation only; test remains untouched during selection.

1. All nine jobs must produce a checkpoint and finite validation history.
2. The unconditioned control must be invariant to QP at its model boundary.
3. QPC must change edit RMS or high-frequency energy by more than 3% between
   QP30 and QP50 while keeping edit RMS below 0.14.
4. Run matched real H.264/H.265 validation evaluation only after the label-free
   conditioning audit passes.
5. Continue QPC only if its three-seed median improves over retrained control by
   at least one BD-rate percentage point on **both codecs**, no seed regresses
   by more than two points, and accuracy guardrails pass.
6. Freeze the winning family and strength before one 1,159-clip test run.

The aspirational success criterion remains BD-rate `<= -15%` on both codecs,
but it is not the continuation threshold and must not be obtained by repeated
test-set selection. If QPC fails the mechanism or matched-control gate, close
scalar QP conditioning. The next credible direction is codec-native rate
allocation/real-codec differentiable training, not another saliency mask grid.

## Reproduction files

- `configs/qpc_v4_ar.yaml`: frozen 16-epoch train configuration.
- `ops/push_qpc_v4.py`: one-job generator/pusher with duplicate protection.
- `kaggle/qpc_v4_train_cell.sh`: Kaggle entry cell without client timeout.
- `ops/gates_qpc.py`: validation-only conditioning audit.
- `ops/audit_kaggle_history.py`: notebook/version inventory.
- `ops/download_kaggle_results.py`: compact result/log downloader.

