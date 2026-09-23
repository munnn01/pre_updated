set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
CODEC="__CODEC__"
REPO="/kaggle/working/pre_updated_codec_search"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/codec_search_ar/${CODEC}"
mkdir -p "$ROOT_OUT"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "codec_search_ar_${CODEC}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  echo "[exit] rc=$rc artifact=/kaggle/working/codec_search_ar_${CODEC}.tgz"
  exit "$rc"
}
trap finish EXIT

git clone -q https://github.com/munnn01/pre_updated.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"
python -c 'import torch, torchvision, cv2; print("torch", torch.__version__, "torchvision", torchvision.__version__, "opencv", cv2.__version__, "cuda", torch.cuda.is_available())'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265'

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

cp docs/RUN_DESIGN_CODEC_SEARCH_AR.md "$ROOT_OUT/preregistered_design.md"
python ops/codec_search_ar.py \
  --index "$INDEX" --codec "$CODEC" --out-dir "$ROOT_OUT" \
  --train-clips 400 --val-clips 104 --preset medium \
  2>&1 | tee "$ROOT_OUT/run.log"

test -f "$ROOT_OUT/calibration.json"
test -f "$ROOT_OUT/search_result.json"
echo "[done] codec search AR codec=$CODEC"
