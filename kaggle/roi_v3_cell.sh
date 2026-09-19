set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
PHASE="__PHASE__"
CODECS="__CODECS__"
EVAL_BACKBONE="__EVAL_BACKBONE__"
N_CLIPS="__N_CLIPS__"
CRFS="__CRFS__"
REPO="/tmp/pre_updated"
OUT="/kaggle/working/outputs/roi_v3"
INDEX="/kaggle/working/kinetics_hash_split.json"

mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  if [ -f "$INDEX" ]; then
    tar -czf roi_v3_outputs.tgz outputs kinetics_hash_split.json
  else
    tar -czf roi_v3_outputs.tgz outputs
  fi
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/roi_v3_outputs.tgz"
  exit "$rc"
}
trap finish EXIT

rm -rf "$REPO"
git clone -q https://github.com/munnn01/pre_updated.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"

python -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available())'
ffmpeg -hide_banner -filters 2>&1 | grep -E ' addroi ' | head -1 || true
ffmpeg -hide_banner -encoders 2>&1 | grep -E 'libx264|libx265' | head

if [ "$PHASE" = "f0" ]; then
  python ops/probe_joint_roi.py \
    --phase f0 --codecs "$CODECS" --f0-crfs 30,38,46 \
    --out "$OUT" 2>&1 | tee "$OUT/run.log"
else
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
    KIN_VIDEO="$(find /kaggle/input -maxdepth 10 -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' \) -print -quit || true)"
    if [ -n "$KIN_VIDEO" ]; then
      KIN_ROOT="/kaggle/input"
    fi
  fi
  if [ -z "$KIN_ROOT" ]; then
    echo "ERROR: D1 requires qktttttttttt/kineticscleaned" >&2
    exit 2
  fi
  echo "[inputs] Kinetics root=$KIN_ROOT"
  python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX"
  python ops/probe_joint_roi.py \
    --phase d1 --index "$INDEX" --split val --n-clips "$N_CLIPS" \
    --num-frames 16 --size 128 --stride 2 --codecs "$CODECS" \
    --crfs "$CRFS" --aq-mode 2 --aq-strength 1 --gop 32 \
    --saliency-teacher r3d_18 --eval-backbone "$EVAL_BACKBONE" \
    --out "$OUT" 2>&1 | tee "$OUT/run.log"
fi

echo "[done] ROI V3 phase=$PHASE codecs=$CODECS evaluator=$EVAL_BACKBONE"
