set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
CODEC="__CODEC__"
SEED="__SEED__"
EXPECTED_BEST_SHA="__EXPECTED_BEST_SHA__"
EXPECTED_LAST_SHA="__EXPECTED_LAST_SHA__"
REPO="/tmp/pre_updated_longtrain_v12"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/longtrain_fullval_v12/${CODEC}"
TRAIN_OUT="$ROOT_OUT/train"
EVAL_OUT="$ROOT_OUT/eval_full1010"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "longtrain_fullval_v12_${CODEC}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/longtrain_fullval_v12_${CODEC}.tgz"
  exit "$rc"
}
trap finish EXIT

rm -rf "$REPO"
git clone -q https://github.com/munnn01/pre_updated.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"

python -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265' || true

KIN_ROOT=""
for candidate in \
  /kaggle/input/kineticscleaned \
  /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then
    KIN_ROOT="$candidate"
    break
  fi
done
if [ -z "$KIN_ROOT" ]; then
  sample="$(find /kaggle/input -maxdepth 10 -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' \) -print -quit || true)"
  [ -n "$sample" ] && KIN_ROOT="$(dirname "$(dirname "$sample")")"
fi
if [ -z "$KIN_ROOT" ]; then
  echo "ERROR: qktttttttttt/kineticscleaned is not mounted" >&2
  exit 2
fi
python scripts/build_train_index.py \
  --root "$KIN_ROOT" --out "$INDEX" --assert-fingerprint 30f083f8520a

case "$CODEC" in
  h264)
    SOURCE_BEST="$(find /kaggle/input -type f \
      -path '*crc-v5-h264-v12-source-s*' -name 'preprocessor.pth' \
      -print -quit || true)"
    SOURCE_LAST="$(find /kaggle/input -type f \
      -path '*crc-v5-h264-v12-source-s*' -name 'preprocessor_last.pth' \
      -print -quit || true)"
    DUAL_LR="0.015"
    ;;
  h265)
    SOURCE_BEST="$(find /kaggle/input -type f \
      -path '*crc-v5-h265-minus24-candidate-v1*' \
      -name 'preprocessor.pth' -print -quit || true)"
    SOURCE_LAST="$SOURCE_BEST"
    DUAL_LR="0.005"
    ;;
  *)
    echo "ERROR: unsupported codec=$CODEC" >&2
    exit 3
    ;;
esac

if [ -z "$SOURCE_BEST" ] || [ -z "$SOURCE_LAST" ]; then
  echo "ERROR: resume checkpoint pair for $CODEC was not found" >&2
  find /kaggle/input -type f -name 'preprocessor*.pth' -print || true
  exit 4
fi
BEST_SHA="$(sha256sum "$SOURCE_BEST" | awk '{print $1}')"
LAST_SHA="$(sha256sum "$SOURCE_LAST" | awk '{print $1}')"
if [ "$BEST_SHA" != "$EXPECTED_BEST_SHA" ]; then
  echo "ERROR: best checkpoint SHA mismatch expected=$EXPECTED_BEST_SHA actual=$BEST_SHA" >&2
  exit 5
fi
if [ "$LAST_SHA" != "$EXPECTED_LAST_SHA" ]; then
  echo "ERROR: last checkpoint SHA mismatch expected=$EXPECTED_LAST_SHA actual=$LAST_SHA" >&2
  exit 6
fi
echo "[source] codec=$CODEC best=$BEST_SHA last=$LAST_SHA seed=$SEED dual_lr=$DUAL_LR"

python - "$SOURCE_LAST" "$CODEC" <<'PY'
import sys

import torch

path, codec = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
assert state["epoch"] == 1, state["epoch"]
assert state["global_step"] == 500, state["global_step"]
assert state.get("opt"), "resume checkpoint has no optimizer state"
cfg = state["cfg"]
assert cfg["model"]["arch"] == "additive_cond"
assert int(cfg["model"]["cond_dim"]) == 3
if codec == "h264":
    assert cfg["codec"]["ste_codec"] == "h264"
    assert cfg["codec"]["ste_alternate"] is False
print(f"[resume-source] codec={codec} epoch=1 step=500 optimizer=present")
PY

mkdir -p "$TRAIN_OUT/checkpoints" "$EVAL_OUT"
cp "$SOURCE_BEST" "$TRAIN_OUT/checkpoints/preprocessor.pth"
cp "$SOURCE_LAST" "$TRAIN_OUT/checkpoints/preprocessor_last.pth"

# The source checkpoint records epoch=1 after a 500-step partial calibration.
# Target epoch=4 therefore executes three additional COMPLETE epochs.  With
# 8,632 train clips and batch size 2 this is about 12,948 added steps.  No
# coefficient, architecture, QP distribution or loss term is changed.
python train.py --config configs/crc_v5_per_codec_ar.yaml \
  data.index="$INDEX" out_dir="$TRAIN_OUT" seed="$SEED" \
  codec.ste_codec="$CODEC" codec.ste_alternate=false \
  train.epochs=4 train.max_steps=null \
  train.resume=true train.finetune=false train.cosine=false train.patience=0 \
  loss.rate_constraint.enabled=true \
  loss.rate_constraint.target_ratio=0.0 \
  loss.rate_constraint.dual_lr="$DUAL_LR" \
  loss.rate_constraint.per_codec."$CODEC".target_ratio=0.0 \
  loss.rate_constraint.per_codec."$CODEC".dual_lr="$DUAL_LR" \
  2>&1 | tee "$TRAIN_OUT/train.log"

FINAL="$TRAIN_OUT/checkpoints/preprocessor_last.pth"
if [ ! -f "$FINAL" ]; then
  echo "ERROR: long training did not produce $FINAL" >&2
  exit 7
fi
FINAL_SHA="$(sha256sum "$FINAL" | awk '{print $1}')"
python - "$FINAL" "$CODEC" "$DUAL_LR" <<'PY'
import math
import sys

import torch

path, codec, dual_lr = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
assert state["epoch"] == 4, state["epoch"]
assert state["global_step"] >= 13000, state["global_step"]
cfg = state["cfg"]
assert cfg["codec"]["ste_codec"] == codec
assert cfg["codec"]["ste_alternate"] is False
assert cfg["train"]["resume"] is True
assert cfg["train"]["finetune"] is False
assert cfg["train"]["max_steps"] is None
rc = cfg["loss"]["rate_constraint"]["per_codec"][codec]
assert math.isclose(float(rc["target_ratio"]), 0.0, abs_tol=1e-12), rc
assert math.isclose(float(rc["dual_lr"]), float(dual_lr), abs_tol=1e-12), rc
duals = state["rate_duals"]
assert set(duals) == {f"{codec}:{qp}" for qp in (30, 35, 40, 45, 50)}, duals
assert all(math.isfinite(float(value)) for value in duals.values())
print(
    f"[longtrain-checkpoint] codec={codec} epoch={state['epoch']} "
    f"step={state['global_step']} duals={duals}"
)
PY
echo "[longtrain] codec=$CODEC final_sha256=$FINAL_SHA"

python evaluate.py --config configs/crc_v5_per_codec_ar.yaml \
  --ckpt "$FINAL" --out "$EVAL_OUT" \
  data.index="$INDEX" eval.split=val eval.codecs="[$CODEC]" \
  eval.shard_idx=0 eval.num_shards=1 eval.shard_salt=longtrain-v12-fullval \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$EVAL_OUT/eval.log"

python - "$FINAL" "$EVAL_OUT/results.json" "$ROOT_OUT/manifest.json" \
  "$CODEC" "$EXPECTED_BEST_SHA" "$EXPECTED_LAST_SHA" "$FINAL_SHA" \
  "$SEED" "$REF" <<'PY'
import json
import math
import sys

import torch

(
    checkpoint_path,
    result_path,
    manifest_path,
    codec,
    source_best_sha,
    source_last_sha,
    final_sha,
    seed,
    commit,
) = sys.argv[1:]
state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
with open(result_path, encoding="utf-8") as handle:
    report = json.load(handle)
assert report["n_eval"] == 1010, report["n_eval"]
metric = report["bd_prep_gain"][f"prep+{codec} vs {codec}"]
assert math.isfinite(float(metric["bd_rate_pct"])), metric
assert math.isfinite(float(metric["bd_accuracy"])), metric
assert report["curves"][codec]["keys"] == [50, 45, 40, 35, 30]
assert report["curves"][f"prep+{codec}"]["keys"] == [50, 45, 40, 35, 30]
manifest = {
    "protocol": "longtrain-fullval-v12",
    "commit": commit,
    "codec": codec,
    "seed": int(seed),
    "source_best_sha256": source_best_sha,
    "source_last_sha256": source_last_sha,
    "final_sha256": final_sha,
    "epoch": int(state["epoch"]),
    "global_step": int(state["global_step"]),
    "n_eval": int(report["n_eval"]),
    "bd_rate_pct": float(metric["bd_rate_pct"]),
    "bd_accuracy": float(metric["bd_accuracy"]),
    "rate_duals": {key: float(value) for key, value in state["rate_duals"].items()},
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
print("[v12-result]", json.dumps(manifest, sort_keys=True))
PY

test "$(sha256sum "$SOURCE_BEST" | awk '{print $1}')" = "$EXPECTED_BEST_SHA"
test "$(sha256sum "$SOURCE_LAST" | awk '{print $1}')" = "$EXPECTED_LAST_SHA"
echo "[done] V12 long-train + full-validation codec=$CODEC n_eval=1010"
