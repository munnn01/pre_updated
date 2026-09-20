set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
ARM="__ARM__"
SEED="__SEED__"
ENABLED="__ENABLED__"
TARGET="__TARGET__"
DUAL_LR="__DUAL_LR__"
BETA="__BETA__"
REPO="/tmp/pre_updated"
INDEX="/kaggle/working/kinetics_hash_split.json"
STAGE1="/kaggle/working/outputs/crc_v5_${ARM}_stage1"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "crc_v5_${ARM}_paired.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/crc_v5_${ARM}_paired.tgz"
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

# Account-local Stage 1. On a corrected notebook version, reuse only this
# kernel's own prior-version checkpoint when it is attached as a kernel source.
mkdir -p "$STAGE1/checkpoints"
PREV_STAGE1="$(find /kaggle/input -type f -path "*/outputs/crc_v5_${ARM}_stage1/checkpoints/preprocessor.pth" -print -quit || true)"
if [ -n "$PREV_STAGE1" ]; then
  cp "$PREV_STAGE1" "$STAGE1/checkpoints/preprocessor.pth"
  echo "[stage1] reused previous-version checkpoint=$PREV_STAGE1"
else
  echo "[stage1] QPC pretrain arm=$ARM seed=$SEED"
  python train.py --config configs/qpc_v4_ar.yaml \
    data.index="$INDEX" out_dir="$STAGE1" seed="$SEED" \
    train.epochs=16 train.resume=false train.finetune=false \
    2>&1 | tee "$STAGE1/train.log"
fi

STAGE1_CKPT="$STAGE1/checkpoints/preprocessor.pth"
if [ ! -f "$STAGE1_CKPT" ]; then
  echo "ERROR: Stage 1 did not produce $STAGE1_CKPT" >&2
  exit 3
fi
STAGE1_SHA="$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')"
echo "[stage1] checkpoint=$STAGE1_CKPT sha256=$STAGE1_SHA"

run_arm() {
  local run_arm="$1"
  local enabled="$2"
  local target="$3"
  local dual_lr="$4"
  local beta="$5"
  local out="/kaggle/working/outputs/crc_v5_${ARM}_${run_arm}"
  local eval_out="$out/eval_val0of20"

  mkdir -p "$out/checkpoints"
  cp "$STAGE1_CKPT" "$out/checkpoints/preprocessor.pth"
  echo "[train] block=$ARM run_arm=$run_arm enabled=$enabled target=$target dual_lr=$dual_lr beta=$beta"
  python train.py --config configs/crc_v5_ar.yaml \
    data.index="$INDEX" out_dir="$out" seed="$SEED" \
    loss.beta="$beta" loss.rate_constraint.enabled="$enabled" \
    loss.rate_constraint.target_ratio="$target" \
    loss.rate_constraint.dual_lr="$dual_lr" \
    2>&1 | tee "$out/train.log"

  local ckpt="$out/checkpoints/preprocessor.pth"
  python - "$ckpt" "$run_arm" "$STAGE1_SHA" <<'PY'
import json
import math
import sys
import torch

path, arm, source_sha = sys.argv[1:]
state = torch.load(path, map_location="cpu")
cfg = state["cfg"]
assert cfg["model"]["cond_dim"] == 3
assert cfg["codec"]["kind"] == "ste" and cfg["codec"]["ste_alternate"] is True
assert state["global_step"] == 500, state["global_step"]
assert all(math.isfinite(float(v)) for v in state.get("rate_duals", {}).values())
print("[checkpoint]", arm, "step=", state["global_step"], "source_sha=", source_sha,
      "duals=", json.dumps(state.get("rate_duals", {}), sort_keys=True))
PY

  python evaluate.py --config configs/crc_v5_ar.yaml \
    --ckpt "$ckpt" --out "$eval_out" \
    data.index="$INDEX" eval.split=val \
    eval.shard_idx=0 eval.num_shards=20 eval.shard_salt=crc-v5-val-v1 \
    eval.per_sequence=true eval.include_proxy=false \
    eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$eval_out.log"
  echo "[done-arm] $run_arm result=$eval_out/results.json"
}

# Within-account paired control: exact same Stage-1 bytes and calibration data.
run_arm control false 0.0 0.0 0.001
run_arm "$ARM" "$ENABLED" "$TARGET" "$DUAL_LR" "$BETA"

echo "[done] paired block=$ARM control+${ARM}"
