set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
SEED="__SEED__"
ARM_ORDER="__ARM_ORDER__"
EXPECTED_SHA="a3580b32e1554071811888238ea249ce9ae26e7b32392a2d975e4bd6cd0a9c8e"
REPO="/tmp/pre_updated_dual_checkpoint_v10"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/dual_checkpoint_v10"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf dual_checkpoint_v10_outputs.tgz outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/dual_checkpoint_v10_outputs.tgz"
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

H265_CKPT="$(find /kaggle/input -type f -path '*crc-v5-h265-minus24-candidate-v1*' -name 'preprocessor.pth' -print -quit || true)"
if [ -z "$H265_CKPT" ]; then
  echo "ERROR: immutable H.265 candidate dataset is not mounted" >&2
  exit 3
fi
ACTUAL_SHA="$(sha256sum "$H265_CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "ERROR: H.265 candidate SHA mismatch expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 4
fi
echo "[source] frozen_h265=$H265_CKPT sha256=$ACTUAL_SHA seed=$SEED order=$ARM_ORDER"

evaluate_h265() {
  local name="$1"
  local shard_idx="$2"
  local num_shards="$3"
  local salt="$4"
  local out="$ROOT_OUT/h265_frozen/$name"
  mkdir -p "$out"
  echo "[h265-eval] name=$name shard=$shard_idx/$num_shards salt=$salt"
  python evaluate.py --config configs/crc_v5_ar.yaml \
    --ckpt "$H265_CKPT" --out "$out" \
    data.index="$INDEX" eval.split=val eval.codecs="[h265]" \
    eval.shard_idx="$shard_idx" eval.num_shards="$num_shards" eval.shard_salt="$salt" \
    eval.per_sequence=true eval.include_proxy=false \
    eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$out.log"
}

# Positive-control reproduction of the historical 44-clip result.  The model is
# mounted read-only and never enters an optimizer.
evaluate_h265 eval_original_val0of20 0 20 crc-v5-val-v1
python - "$ROOT_OUT/h265_frozen/eval_original_val0of20/results.json" <<'PY'
import json, math, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
metric = report["bd_prep_gain"]["prep+h265 vs h265"]
assert report["n_eval"] == 44, report["n_eval"]
assert math.isclose(metric["bd_rate_pct"], -24.2623162235382, abs_tol=0.25), metric
assert math.isclose(metric["bd_accuracy"], 0.0318284840950696, abs_tol=0.01), metric
print("[h265-reproduced]", json.dumps(metric, sort_keys=True), "n_eval=44")
PY

# A locked unseen shard checks whether the candidate transfers beyond the shard
# that produced -24.26%.  It is descriptive and does not alter the checkpoint.
evaluate_h265 eval_fresh_val2of20 2 20 dual-checkpoint-v10-val-v1

run_h264_arm() {
  local arm="$1"
  local target="$2"
  local dual_lr="$3"
  local out="$ROOT_OUT/h264/$arm"
  local eval_out="$out/eval_fresh_val2of20"

  mkdir -p "$out/checkpoints" "$eval_out"
  cp "$H265_CKPT" "$out/checkpoints/preprocessor.pth"
  echo "[h264-train] arm=$arm target=$target dual_lr=$dual_lr source_sha=$ACTUAL_SHA"
  python train.py --config configs/crc_v5_per_codec_ar.yaml \
    data.index="$INDEX" out_dir="$out" seed="$SEED" \
    codec.ste_codec=h264 codec.ste_alternate=false \
    train.balanced_codec_qp=true train.max_steps=500 \
    loss.rate_constraint.enabled=true \
    loss.rate_constraint.target_ratio="$target" \
    loss.rate_constraint.dual_lr="$dual_lr" \
    loss.rate_constraint.per_codec.h264.target_ratio="$target" \
    loss.rate_constraint.per_codec.h264.dual_lr="$dual_lr" \
    2>&1 | tee "$out/train.log"

  local ckpt="$out/checkpoints/preprocessor.pth"
  python - "$ckpt" "$arm" "$target" "$dual_lr" "$ACTUAL_SHA" <<'PY'
import json, math, sys, torch
path, arm, target, dual_lr, source_sha = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
cfg = state["cfg"]
rc = cfg["loss"]["rate_constraint"]["per_codec"]["h264"]
assert cfg["codec"]["ste_codec"] == "h264"
assert cfg["codec"]["ste_alternate"] is False
assert state["global_step"] == 500, state["global_step"]
assert float(rc["target_ratio"]) == float(target), rc
assert float(rc["dual_lr"]) == float(dual_lr), rc
duals = state.get("rate_duals", {})
assert set(duals) == {f"h264:{qp}" for qp in (30, 35, 40, 45, 50)}, duals
assert all(math.isfinite(float(value)) for value in duals.values())
print("[h264-checkpoint]", arm, "step=500 source_sha=", source_sha,
      "duals=", json.dumps(duals, sort_keys=True))
PY

  python evaluate.py --config configs/crc_v5_per_codec_ar.yaml \
    --ckpt "$ckpt" --out "$eval_out" \
    data.index="$INDEX" eval.split=val eval.codecs="[h264]" \
    eval.shard_idx=2 eval.num_shards=20 eval.shard_salt=dual-checkpoint-v10-val-v1 \
    eval.per_sequence=true eval.include_proxy=false \
    eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$eval_out.log"
  echo "[done-h264-arm] $arm result=$eval_out/results.json"

  # The mounted H.265 checkpoint must remain byte-identical after every H.264 arm.
  test "$(sha256sum "$H265_CKPT" | awk '{print $1}')" = "$EXPECTED_SHA"
}

IFS=',' read -r -a ARMS <<< "$ARM_ORDER"
if [ "${#ARMS[@]}" -ne 5 ]; then
  echo "ERROR: arm order must contain exactly five arms" >&2
  exit 5
fi
for arm in "${ARMS[@]}"; do
  case "$arm" in
    t0_lr5)     run_h264_arm "$arm" 0.0 0.005 ;;
    t0_lr15)    run_h264_arm "$arm" 0.0 0.015 ;;
    tm5_lr5)    run_h264_arm "$arm" -0.05 0.005 ;;
    tm5_lr15)   run_h264_arm "$arm" -0.05 0.015 ;;
    tm25_lr10)  run_h264_arm "$arm" -0.025 0.010 ;;
    *)
      echo "ERROR: unknown H.264 arm: $arm" >&2
      exit 6
      ;;
  esac
done

echo "[done] frozen H.265 + independent H.264 factorial complete"
