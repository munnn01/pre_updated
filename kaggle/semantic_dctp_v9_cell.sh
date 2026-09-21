set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
CODEC="__CODEC__"
RATE_ARM="__RATE_ARM__"
EXPECTED_SHA="__EXPECTED_SHA__"
TEMPORAL="__TEMPORAL__"
ARMS="p25_q1:0.25:1.0 p25_q2:0.25:2.0 p25_q4:0.25:4.0 p50_q1:0.50:1.0 p50_q2:0.50:2.0 p50_q4:0.50:4.0 p375_q2:0.375:2.0"
REPO="/tmp/pre_updated"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/semantic_dctp_v9_${CODEC}"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "semantic_dctp_v9_${CODEC}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/semantic_dctp_v9_${CODEC}.tgz"
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

SOURCE="$(find /kaggle/input -type f \
  -path "*codec_specific_v7_${CODEC}_${RATE_ARM}_${RATE_ARM}/checkpoints/preprocessor.pth" \
  -print -quit || true)"
if [ -z "$SOURCE" ]; then
  echo "ERROR: treatment checkpoint from attached V7 kernel was not found" >&2
  find /kaggle/input -type f -name 'preprocessor.pth' -print || true
  exit 3
fi
ACTUAL_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "ERROR: V7 checkpoint SHA mismatch expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 4
fi
echo "[source] codec=$CODEC rate_arm=$RATE_ARM checkpoint=$SOURCE sha256=$ACTUAL_SHA"

BASE_OUT="$ROOT_OUT/baseline/eval_val0of10"
mkdir -p "$BASE_OUT"
python evaluate.py --config configs/crc_v5_ar.yaml \
  --ckpt "$SOURCE" --out "$BASE_OUT" \
  data.index="$INDEX" eval.split=val eval.codecs="[$CODEC]" \
  eval.shard_idx=0 eval.num_shards=10 eval.shard_salt=semantic-dctp-v9-val-v1 \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$ROOT_OUT/baseline.log"

for SPEC in $ARMS; do
  IFS=: read -r TAG PROTECT THRESHOLD <<< "$SPEC"
  OUT="$ROOT_OUT/$TAG"
  CKPT="$OUT/checkpoints/preprocessor.pth"
  EVAL_OUT="$OUT/eval_val0of10"
  mkdir -p "$OUT/checkpoints"
  python ops/make_dctp_v6_ckpt.py \
    --source "$SOURCE" --out "$CKPT" --expected-sha "$EXPECTED_SHA" \
    --residual-scale 0.0 --dct-strength 1.0 \
    --dct-threshold "$THRESHOLD" --dct-band-start 2 \
    --temporal-strength "$TEMPORAL" --qp-slope 0.75 \
    --semantic-protect-area "$PROTECT" --seed 240921
  echo "[semantic-compose] codec=$CODEC arm=$TAG protect=$PROTECT threshold=$THRESHOLD temporal=$TEMPORAL"
  python evaluate.py --config configs/semantic_dctp_v9_ar.yaml \
    --ckpt "$CKPT" --out "$EVAL_OUT" \
    data.index="$INDEX" eval.split=val eval.codecs="[$CODEC]" \
    eval.shard_idx=0 eval.num_shards=10 eval.shard_salt=semantic-dctp-v9-val-v1 \
    eval.per_sequence=true eval.include_proxy=false \
    eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$OUT/eval.log"
done

echo "[done] Semantic-DCTP V9 codec=$CODEC arms=$ARMS"
