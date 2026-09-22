set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
SEED="__SEED__"
REPO="/tmp/pre_updated_additive_attn"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/additive_attn_ar/seed${SEED}"
TRAIN_OUT="$ROOT_OUT/train"
EVAL_OUT="$ROOT_OUT/eval"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "additive_attn_ar_seed${SEED}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/additive_attn_ar_seed${SEED}.tgz"
  exit "$rc"
}
trap finish EXIT

rm -rf "$REPO"
# Clone from GitHub with commit pin; if git fails, fall back to dataset
if git clone -q https://github.com/munnn01/pre_updated.git "$REPO"; then
  git -C "$REPO" checkout -q "$REF"
elif [ -d "/kaggle/input/pre-updated-copy" ]; then
  echo "[source] using /kaggle/input/pre-updated-copy fallback"
  mkdir -p "$REPO"
  cp -r /kaggle/input/pre-updated-copy/* "$REPO/"
elif [ -d "/kaggle/input/datasets/qktttttttttt/pre-updated-copy" ]; then
  echo "[source] using /kaggle/input/datasets/qktttttttttt/pre-updated-copy fallback"
  mkdir -p "$REPO"
  cp -r /kaggle/input/datasets/qktttttttttt/pre-updated-copy/* "$REPO/"
else
  echo "ERROR: could not clone or locate repository" >&2
  exit 1
fi
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

mkdir -p "$ROOT_OUT"

echo "[train] starting training additive_attn_ar seed=$SEED..."
python train.py --config configs/additive_attn_ar.yaml \
  data.index="$INDEX" \
  seed="$SEED" \
  out_dir="$ROOT_OUT" \
  train.resume=true

echo "[train] training completed, searching for checkpoint..."
CKPT="$(find "$ROOT_OUT/checkpoints" -name "preprocessor.pth" -print -quit || true)"
if [ -z "$CKPT" ]; then
  CKPT="$(find "$ROOT_OUT/checkpoints" -name "preprocessor_last.pth" -print -quit || true)"
fi
if [ -z "$CKPT" ]; then
  echo "ERROR: no checkpoint found in $ROOT_OUT/checkpoints" >&2
  ls -la "$ROOT_OUT/checkpoints" || true
  exit 3
fi

echo "[eval] running evaluate with checkpoint: $CKPT"
python evaluate.py --config configs/additive_attn_ar.yaml \
  --ckpt "$CKPT" \
  --out "$EVAL_OUT" \
  data.index="$INDEX" \
  seed="$SEED"

echo "[done] seed=$SEED finished successfully"
