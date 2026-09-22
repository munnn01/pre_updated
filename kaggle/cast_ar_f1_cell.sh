set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
CODEC="__CODEC__"
SEED="__SEED__"
REPO="/tmp/pre_updated_cast_ar_f1"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/cast_ar_f1/${CODEC}_seed${SEED}"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "cast_ar_f1_${CODEC}_seed${SEED}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/cast_ar_f1_${CODEC}_seed${SEED}.tgz"
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
mkdir -p "$ROOT_OUT"
cp docs/RUN_DESIGN_CAST_AR_F1.md "$ROOT_OUT/preregistered_design.md"

python ops/cast_ar_post_screen.py \
  --index "$INDEX" \
  --codec "$CODEC" \
  --seed "$SEED" \
  --out-dir "$ROOT_OUT" \
  --preset medium \
  --train-clips 1200 \
  --val-clips 208 \
  --val-salt cast-ar-f1-val-v1 \
  --epochs 3 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --workers 2 \
  --width 24 \
  --bases 4 \
  --max-delta 0.15 \
  --lr 3e-4 \
  2>&1 | tee "$ROOT_OUT/run.log"

test -f "$ROOT_OUT/cast_ar_post.pth"
test -f "$ROOT_OUT/screen_result.json"
echo "[done] CAST-AR F1 codec=$CODEC seed=$SEED"

