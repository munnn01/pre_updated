set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
ARM="__ARM__"
ARCH="__ARCH__"
SEED="__SEED__"
EPOCHS="__EPOCHS__"
QP_WEIGHTS="__QP_WEIGHTS__"
REPO="/tmp/pre_updated"
OUT="/kaggle/working/outputs/qpc_v4_${ARM}_s${SEED}"
INDEX="/kaggle/working/kinetics_hash_split.json"

mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "qpc_v4_${ARM}_s${SEED}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/qpc_v4_${ARM}_s${SEED}.tgz"
  exit "$rc"
}
trap finish EXIT

rm -rf "$REPO"
git clone -q https://github.com/munnn01/pre_updated.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"

python -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'

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

echo "[train] arm=$ARM arch=$ARCH seed=$SEED epochs=$EPOCHS qp_weights=$QP_WEIGHTS"
python train.py --config configs/qpc_v4_ar.yaml \
  data.index="$INDEX" out_dir="$OUT" seed="$SEED" \
  model.arch="$ARCH" train.epochs="$EPOCHS" \
  train.qp_sampling_weights="$QP_WEIGHTS" \
  2>&1 | tee "$OUT/train.log"

CKPT="$OUT/checkpoints/preprocessor.pth"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: training did not produce $CKPT" >&2
  exit 3
fi

# A gate failure is a scientific result, not a broken Kaggle execution. Keep
# the kernel COMPLETE and persist the audit JSON/exit code for selection.
set +e
python ops/gates_qpc.py \
  --ckpt "$CKPT" --index "$INDEX" --split val --n-clips 32 \
  --out "$OUT/conditioning_audit.json" \
  2>&1 | tee "$OUT/conditioning_audit.log"
AUDIT_RC=${PIPESTATUS[0]}
set -e
echo "$AUDIT_RC" > "$OUT/conditioning_audit.rc"
echo "[done] arm=$ARM seed=$SEED audit_rc=$AUDIT_RC"

