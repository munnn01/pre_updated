set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
EXPECTED_SHA="__EXPECTED_SHA__"
CODEC="__CODEC__"
ARM="__ARM__"
TARGET="__TARGET__"
DUAL_LR="__DUAL_LR__"
SEED="__SEED__"
REPO="/tmp/pre_updated"
INDEX="/kaggle/working/kinetics_hash_split.json"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "codec_specific_v7_${CODEC}_${ARM}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/codec_specific_v7_${CODEC}_${ARM}.tgz"
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

STAGE1_CKPT="$(find /kaggle/input -type f -name 'preprocessor.pth' -print -quit || true)"
if [ -z "$STAGE1_CKPT" ]; then
  echo "ERROR: canonical dieulinhh Stage-1 dataset lacks preprocessor.pth" >&2
  exit 3
fi
ACTUAL_SHA="$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "ERROR: Stage-1 SHA mismatch expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 4
fi
echo "[source] checkpoint=$STAGE1_CKPT sha256=$ACTUAL_SHA codec=$CODEC arm=$ARM"

run_arm() {
  local run_arm="$1"
  local enabled="$2"
  local target="$3"
  local dual_lr="$4"
  local beta="$5"
  local out="/kaggle/working/outputs/codec_specific_v7_${CODEC}_${ARM}_${run_arm}"
  local eval_out="$out/eval_val0of10"

  mkdir -p "$out/checkpoints"
  cp "$STAGE1_CKPT" "$out/checkpoints/preprocessor.pth"
  echo "[train] codec=$CODEC run_arm=$run_arm enabled=$enabled target=$target dual_lr=$dual_lr beta=$beta"
  python train.py --config configs/crc_v5_ar.yaml \
    data.index="$INDEX" out_dir="$out" seed="$SEED" \
    codec.ste_codec="$CODEC" codec.ste_alternate=false \
    train.balanced_codec_qp=true \
    loss.beta="$beta" loss.rate_constraint.enabled="$enabled" \
    loss.rate_constraint.target_ratio="$target" \
    loss.rate_constraint.dual_lr="$dual_lr" \
    2>&1 | tee "$out/train.log"

  echo "[eval] codec=$CODEC run_arm=$run_arm"
  python evaluate.py --config configs/crc_v5_ar.yaml \
    --ckpt "$out/checkpoints/preprocessor.pth" --out "$eval_out" \
    data.index="$INDEX" eval.split=val eval.codecs="[$CODEC]" \
    eval.shard_idx=0 eval.num_shards=10 eval.shard_salt=codec-specific-v7-val-v1 \
    eval.per_sequence=true eval.include_proxy=false \
    eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$eval_out.log"
  echo "[done-arm] $run_arm result=$eval_out/results.json"
}

# Both branches fork the exact same immutable Stage-1 bytes. The fixed-beta
# branch is the concurrent control; the constrained branch changes only the
# registered codec-specific rate treatment.
run_arm control false 0.0 0.0 0.001
run_arm "$ARM" true "$TARGET" "$DUAL_LR" 0.0
echo "[done] codec-specific V7 codec=$CODEC arm=$ARM"
