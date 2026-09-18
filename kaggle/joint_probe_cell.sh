set -euo pipefail
export PYTHONUNBUFFERED=1

# Change to "confirmatory" for the preregistered, publication-size run.
PROFILE="${PROFILE:-quick}"
REF="__REF__"
REPO="/tmp/pre_updated"
OUT="/kaggle/working/outputs"
INDEX="/kaggle/working/kinetics_hash_split.json"
RUN_AR="${RUN_AR:-1}"
RUN_OD="${RUN_OD:-1}"

if [ "$PROFILE" = "confirmatory" ]; then
  N_OD="${N_OD:-500}"
  N_AR="${N_AR:-200}"
  QPS="${QPS:-30,35,40,45,50}"
  OD_SIGMAS="${OD_SIGMAS:-4,8,16}"
  OD_SCORE="${OD_SCORE:-0.5}"
  OD_DILATE="${OD_DILATE:-0.15}"
  OD_FEATHER="${OD_FEATHER:-4}"
  OD_ROI_SIGMAS="${OD_ROI_SIGMAS:-0}"
  OD_POST_SIGMAS="${OD_POST_SIGMAS:-0}"
  OD_POST_MIN_QP="${OD_POST_MIN_QP:-40}"
  OD_MIN_MARGIN_PX="${OD_MIN_MARGIN_PX:-0}"
  OD_MASK_GRID="${OD_MASK_GRID:-1}"
  OD_SEED="${OD_SEED:-0}"
  OD_CODECS="${OD_CODECS:-h264,h265}"
  AR_SIGMAS="${AR_SIGMAS:-4,8}"
  TEMPORAL_STRENGTHS="${TEMPORAL_STRENGTHS:-0,0.5}"
  BOOTSTRAP="${BOOTSTRAP:-1000}"
else
  N_OD="${N_OD:-50}"
  N_AR="${N_AR:-20}"
  QPS="${QPS:-35,40,45}"
  OD_SIGMAS="${OD_SIGMAS:-2,4}"
  OD_SCORE="${OD_SCORE:-0.2}"
  OD_DILATE="${OD_DILATE:-0.30}"
  OD_FEATHER="${OD_FEATHER:-8}"
  OD_ROI_SIGMAS="${OD_ROI_SIGMAS:-0}"
  OD_POST_SIGMAS="${OD_POST_SIGMAS:-0}"
  OD_POST_MIN_QP="${OD_POST_MIN_QP:-40}"
  OD_MIN_MARGIN_PX="${OD_MIN_MARGIN_PX:-0}"
  OD_MASK_GRID="${OD_MASK_GRID:-1}"
  OD_SEED="${OD_SEED:-0}"
  OD_CODECS="${OD_CODECS:-h264,h265}"
  AR_SIGMAS="${AR_SIGMAS:-4}"
  TEMPORAL_STRENGTHS="${TEMPORAL_STRENGTHS:-0.25}"
  BOOTSTRAP="${BOOTSTRAP:-0}"
fi

mkdir -p "$OUT"
finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf joint_probe_outputs.tgz outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/joint_probe_outputs.tgz"
  exit "$rc"
}
trap finish EXIT

rm -rf "$REPO"
git clone -q https://github.com/munnn01/pre_updated.git "$REPO"
git -C "$REPO" checkout -q "$REF"
cd "$REPO"

python -m pip install -q pycocotools
python -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.cuda.is_available())'
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265' | head

# Support both Kaggle's classic and its newer owner/slug mount layouts.  Inputs
# are task-conditional: an OD-only notebook must not fail because Kinetics was
# deliberately not attached, and the symmetric rule applies to AR-only runs.
COCO_VAL=""
COCO_ANN=""
if [ "$RUN_OD" = "1" ]; then
  COCO_VAL="$(find /kaggle/input -maxdepth 10 -type d -name val2017 -print -quit || true)"
  COCO_ANN="$(find /kaggle/input -maxdepth 10 -type f -name instances_val2017.json -print -quit || true)"
fi
KIN_ROOT=""
if [ "$RUN_AR" = "1" ]; then
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
fi

printf '[inputs] COCO val=%s\n[inputs] COCO ann=%s\n[inputs] Kinetics root=%s\n' \
  "$COCO_VAL" "$COCO_ANN" "$KIN_ROOT"
if [ "$RUN_OD" = "1" ] && { [ -z "$COCO_VAL" ] || [ -z "$COCO_ANN" ]; }; then
  echo "ERROR: RUN_OD=1 requires awsaf49/coco-2017-dataset" >&2
  exit 2
fi
if [ "$RUN_AR" = "1" ] && [ -z "$KIN_ROOT" ]; then
  echo "ERROR: RUN_AR=1 requires qktttttttttt/kineticscleaned" >&2
  exit 2
fi
if [ "$RUN_AR" = "1" ]; then
  python scripts/build_train_index.py --root "$KIN_ROOT" --out "$INDEX"
fi

if [ "$RUN_AR" = "1" ]; then
  echo "[stage] AR importance-tube start"
  python ops/probe_action_tubes.py \
    --index "$INDEX" --split test --n-clips "$N_AR" \
    --num-frames 16 --size 128 --qps "$QPS" \
    --sigmas "$AR_SIGMAS" --temporal-strengths "$TEMPORAL_STRENGTHS" \
    --score 0.2 --dilate 0.30 --feather 8 --ar-backbone r3d_18 \
    --out "$OUT/ar_importance_tubes" 2>&1 | tee "$OUT/ar_importance_tubes.log"
  echo "[stage] AR importance-tube complete"
fi

if [ "$RUN_OD" = "1" ]; then
  echo "[stage] OD background-suppression start"
  python ops/probe_background_suppression.py \
    --images "$COCO_VAL" --ann "$COCO_ANN" \
    --n-images "$N_OD" --size 320 --qps "$QPS" \
    --sigmas "$OD_SIGMAS" --score "$OD_SCORE" --dilate "$OD_DILATE" \
    --feather "$OD_FEATHER" --roi-sigmas "$OD_ROI_SIGMAS" \
    --post-sigmas "$OD_POST_SIGMAS" --post-min-qp "$OD_POST_MIN_QP" \
    --min-margin-px "$OD_MIN_MARGIN_PX" --mask-grid "$OD_MASK_GRID" \
    --bootstrap "$BOOTSTRAP" --seed "$OD_SEED" --codecs "$OD_CODECS" \
    --out "$OUT/od_background_suppression" 2>&1 | tee "$OUT/od_background_suppression.log"
  echo "[stage] OD background-suppression complete"
fi

echo "[done] selected probes complete (AR=$RUN_AR OD=$RUN_OD)"
