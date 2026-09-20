set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
ARM="__ARM__"
ARCH="__ARCH__"
SEED="__SEED__"
SHARD_IDX="__SHARD_IDX__"
NUM_SHARDS="__NUM_SHARDS__"
REPO="/tmp/pre_updated"
OUT="/kaggle/working/outputs/qpc_v4_eval_${ARM}_s${SEED}_val${SHARD_IDX}of${NUM_SHARDS}"
INDEX="/kaggle/working/kinetics_hash_split.json"

mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "qpc_v4_eval_${ARM}_s${SEED}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/qpc_v4_eval_${ARM}_s${SEED}.tgz"
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
  if [ -n "$sample" ]; then
    KIN_ROOT="$(dirname "$(dirname "$sample")")"
  fi
fi
if [ -z "$KIN_ROOT" ]; then
  echo "ERROR: qktttttttttt/kineticscleaned is not mounted" >&2
  exit 2
fi
echo "[inputs] Kinetics root=$KIN_ROOT"
python scripts/build_train_index.py \
  --root "$KIN_ROOT" --out "$INDEX" --assert-fingerprint 30f083f8520a

CKPT="$(find /kaggle/input -type f -path "*/outputs/qpc_v4_${ARM}_s${SEED}/checkpoints/preprocessor.pth" -print -quit || true)"
if [ -z "$CKPT" ]; then
  echo "ERROR: attached train output lacks qpc_v4_${ARM}_s${SEED}/checkpoints/preprocessor.pth" >&2
  find /kaggle/input -type f -name 'preprocessor.pth' -print || true
  exit 3
fi
echo "[inputs] checkpoint=$CKPT"
python - "$CKPT" "$ARCH" "$SEED" <<'PY'
import sys
import torch

path, expected_arch, expected_seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
state = torch.load(path, map_location="cpu")
cfg = state.get("cfg", {})
assert cfg.get("model", {}).get("arch") == expected_arch, cfg.get("model")
assert int(cfg.get("seed", -1)) == expected_seed, cfg.get("seed")
assert int(state.get("epoch", -1)) == 14, state.get("epoch")
print(
    "[checkpoint]",
    "arch=", expected_arch,
    "seed=", expected_seed,
    "best_epoch=", state.get("epoch"),
    "best_val=", state.get("best_val"),
)
PY

echo "[eval] arm=$ARM seed=$SEED split=val shard=$SHARD_IDX/$NUM_SHARDS"
python evaluate.py --config configs/qpc_v4_ar.yaml \
  --ckpt "$CKPT" --out "$OUT" \
  data.index="$INDEX" eval.split=val \
  eval.shard_idx="$SHARD_IDX" eval.num_shards="$NUM_SHARDS" \
  eval.shard_salt=qpc-v4-val-v1 \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$OUT/eval.log"

echo "[done] arm=$ARM seed=$SEED result=$OUT/results.json"
