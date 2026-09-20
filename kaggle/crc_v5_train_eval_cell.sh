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
OUT="/kaggle/working/outputs/crc_v5_${ARM}"
INDEX="/kaggle/working/kinetics_hash_split.json"
EXPECTED_WARM_SHA="20d83d69f8be9e6d7754e07fe3c493e46a0df17d8734954ed7d827e0ec372db2"

mkdir -p "$OUT/checkpoints"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "crc_v5_${ARM}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/crc_v5_${ARM}.tgz"
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

WARM="$(find /kaggle/input -type f -name 'preprocessor.pth' -print -quit || true)"
if [ -z "$WARM" ]; then
  echo "ERROR: warm-start dataset lacks preprocessor.pth" >&2
  exit 3
fi
ACTUAL_WARM_SHA="$(sha256sum "$WARM" | awk '{print $1}')"
if [ "$ACTUAL_WARM_SHA" != "$EXPECTED_WARM_SHA" ]; then
  echo "ERROR: warm-start SHA mismatch: $ACTUAL_WARM_SHA" >&2
  exit 4
fi
cp "$WARM" "$OUT/checkpoints/preprocessor.pth"
echo "[warmstart] $WARM sha256=$ACTUAL_WARM_SHA"

echo "[train] arm=$ARM enabled=$ENABLED target=$TARGET dual_lr=$DUAL_LR beta=$BETA"
python train.py --config configs/crc_v5_ar.yaml \
  data.index="$INDEX" out_dir="$OUT" seed="$SEED" \
  loss.beta="$BETA" loss.rate_constraint.enabled="$ENABLED" \
  loss.rate_constraint.target_ratio="$TARGET" \
  loss.rate_constraint.dual_lr="$DUAL_LR" \
  2>&1 | tee "$OUT/train.log"

CKPT="$OUT/checkpoints/preprocessor.pth"
python - "$CKPT" "$ARM" <<'PY'
import json
import math
import sys
import torch

path, arm = sys.argv[1:]
state = torch.load(path, map_location="cpu")
cfg = state["cfg"]
assert cfg["model"]["cond_dim"] == 3
assert cfg["codec"]["kind"] == "ste" and cfg["codec"]["ste_alternate"] is True
assert state["global_step"] == 500, state["global_step"]
assert all(math.isfinite(float(v)) for v in state.get("rate_duals", {}).values())
print("[checkpoint]", arm, "step=", state["global_step"], "duals=", json.dumps(state.get("rate_duals", {}), sort_keys=True))
PY

EVAL_OUT="$OUT/eval_val0of20"
python evaluate.py --config configs/crc_v5_ar.yaml \
  --ckpt "$CKPT" --out "$EVAL_OUT" \
  data.index="$INDEX" eval.split=val \
  eval.shard_idx=0 eval.num_shards=20 eval.shard_salt=crc-v5-val-v1 \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$EVAL_OUT.log"

echo "[done] arm=$ARM result=$EVAL_OUT/results.json"

