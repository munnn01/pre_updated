set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
ARM="__ARM__"
EXPECTED_SHA="__EXPECTED_SHA__"
RESIDUAL="__RESIDUAL__"
DCT="__DCT__"
TEMPORAL="__TEMPORAL__"
REPO="/tmp/pre_updated"
INDEX="/kaggle/working/kinetics_hash_split.json"
OUT="/kaggle/working/outputs/dctp_v6_${ARM}"
CKPT="$OUT/checkpoints/preprocessor.pth"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "dctp_v6_${ARM}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/dctp_v6_${ARM}.tgz"
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

SOURCE="$(find /kaggle/input -type f -name 'preprocessor.pth' -print -quit || true)"
if [ -z "$SOURCE" ]; then
  echo "ERROR: public canonical checkpoint is not mounted" >&2
  exit 3
fi
mkdir -p "$OUT/checkpoints"
python ops/make_dctp_v6_ckpt.py \
  --source "$SOURCE" --out "$CKPT" --expected-sha "$EXPECTED_SHA" \
  --residual-scale "$RESIDUAL" --dct-strength "$DCT" \
  --dct-threshold 1.0 --temporal-strength "$TEMPORAL" --seed 220921

echo "[eval] arm=$ARM residual=$RESIDUAL dct=$DCT temporal=$TEMPORAL"
python evaluate.py --config configs/dctp_v6_ar.yaml \
  --ckpt "$CKPT" --out "$OUT/eval_newval0of10" \
  data.index="$INDEX" eval.split=val \
  eval.shard_idx=0 eval.num_shards=10 eval.shard_salt=dctp-v6-screen-v1 \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$OUT/eval.log"
echo "[done] result=$OUT/eval_newval0of10/results.json"
