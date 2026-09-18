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
  OD_SIZE="${OD_SIZE:-320}"
  QPS="${QPS:-30,35,40,45,50}"
  OD_SIGMAS="${OD_SIGMAS:-4,8,16}"
  OD_SCORE="${OD_SCORE:-0.5}"
  OD_DILATE="${OD_DILATE:-0.15}"
  OD_FEATHER="${OD_FEATHER:-4}"
  OD_ROI_SIGMAS="${OD_ROI_SIGMAS:-0}"
  OD_POST_SIGMAS="${OD_POST_SIGMAS:-0}"
  OD_POST_MIN_QP="${OD_POST_MIN_QP:-45}"
  OD_MIN_MARGIN_PX="${OD_MIN_MARGIN_PX:-0}"
  OD_MASK_GRID="${OD_MASK_GRID:-1}"
  OD_SEED="${OD_SEED:-0}"
  OD_CODECS="${OD_CODECS:-h264,h265}"
  OD_MASK_BACKBONE="${OD_MASK_BACKBONE:-fasterrcnn_mobilenet_v3_large_fpn}"
  OD_EVAL_BACKBONE="${OD_EVAL_BACKBONE:-fasterrcnn_resnet50_fpn}"
  AR_SIGMAS="${AR_SIGMAS:-4,8}"
  TEMPORAL_STRENGTHS="${TEMPORAL_STRENGTHS:-0,0.5}"
  AR_PROBE="${AR_PROBE:-tubes}"
  AR_SPLIT="${AR_SPLIT:-test}"
  AR_BACKBONE="${AR_BACKBONE:-r3d_18}"
  AR_POST_SIGMAS="${AR_POST_SIGMAS:-1}"
  AR_POST_MIN_QP="${AR_POST_MIN_QP:-45}"
  AR_MOTION_QUANTILES="${AR_MOTION_QUANTILES:-0.5,0.75}"
  AR_MOTION_SIGMA="${AR_MOTION_SIGMA:-1}"
  AR_MOTION_DILATION="${AR_MOTION_DILATION:-2}"
  AR_MOTION_FEATHER="${AR_MOTION_FEATHER:-2}"
  AR_SALIENCY_TEACHER="${AR_SALIENCY_TEACHER:-r3d_18}"
  AR_PROTECT_FRACTIONS="${AR_PROTECT_FRACTIONS:-0.15,0.25,0.4}"
  AR_SALIENCY_MODES="${AR_SALIENCY_MODES:-clip,tube}"
  AR_SALIENCY_SIGMA="${AR_SALIENCY_SIGMA:-8}"
  AR_TEMPORAL_STRENGTH="${AR_TEMPORAL_STRENGTH:-0.75}"
  BOOTSTRAP="${BOOTSTRAP:-1000}"
else
  N_OD="${N_OD:-50}"
  N_AR="${N_AR:-20}"
  OD_SIZE="${OD_SIZE:-320}"
  QPS="${QPS:-35,40,45}"
  OD_SIGMAS="${OD_SIGMAS:-2,4}"
  OD_SCORE="${OD_SCORE:-0.2}"
  OD_DILATE="${OD_DILATE:-0.30}"
  OD_FEATHER="${OD_FEATHER:-8}"
  OD_ROI_SIGMAS="${OD_ROI_SIGMAS:-0}"
  OD_POST_SIGMAS="${OD_POST_SIGMAS:-0}"
  OD_POST_MIN_QP="${OD_POST_MIN_QP:-45}"
  OD_MIN_MARGIN_PX="${OD_MIN_MARGIN_PX:-0}"
  OD_MASK_GRID="${OD_MASK_GRID:-1}"
  OD_SEED="${OD_SEED:-0}"
  OD_CODECS="${OD_CODECS:-h264,h265}"
  OD_MASK_BACKBONE="${OD_MASK_BACKBONE:-fasterrcnn_mobilenet_v3_large_fpn}"
  OD_EVAL_BACKBONE="${OD_EVAL_BACKBONE:-fasterrcnn_resnet50_fpn}"
  AR_SIGMAS="${AR_SIGMAS:-4}"
  TEMPORAL_STRENGTHS="${TEMPORAL_STRENGTHS:-0.25}"
  AR_PROBE="${AR_PROBE:-tubes}"
  AR_SPLIT="${AR_SPLIT:-test}"
  AR_BACKBONE="${AR_BACKBONE:-r3d_18}"
  AR_POST_SIGMAS="${AR_POST_SIGMAS:-1}"
  AR_POST_MIN_QP="${AR_POST_MIN_QP:-45}"
  AR_MOTION_QUANTILES="${AR_MOTION_QUANTILES:-0.5,0.75}"
  AR_MOTION_SIGMA="${AR_MOTION_SIGMA:-1}"
  AR_MOTION_DILATION="${AR_MOTION_DILATION:-2}"
  AR_MOTION_FEATHER="${AR_MOTION_FEATHER:-2}"
  AR_SALIENCY_TEACHER="${AR_SALIENCY_TEACHER:-r3d_18}"
  AR_PROTECT_FRACTIONS="${AR_PROTECT_FRACTIONS:-0.15,0.25,0.4}"
  AR_SALIENCY_MODES="${AR_SALIENCY_MODES:-clip,tube}"
  AR_SALIENCY_SIGMA="${AR_SALIENCY_SIGMA:-8}"
  AR_TEMPORAL_STRENGTH="${AR_TEMPORAL_STRENGTH:-0.75}"
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
  if [ "$AR_PROBE" = "post" ]; then
    echo "[stage] AR decoder-POST start"
    python ops/probe_action_post.py \
      --index "$INDEX" --split "$AR_SPLIT" --n-clips "$N_AR" \
      --num-frames 16 --size 128 --qps "$QPS" \
      --post-sigmas "$AR_POST_SIGMAS" --post-min-qp "$AR_POST_MIN_QP" \
      --ar-backbone "$AR_BACKBONE" --out "$OUT/ar_decoder_post" \
      2>&1 | tee "$OUT/ar_decoder_post.log"
    echo "[stage] AR decoder-POST complete"
  elif [ "$AR_PROBE" = "motion" ]; then
    echo "[stage] AR motion-preserving POST start"
    python ops/probe_action_motion_post.py \
      --index "$INDEX" --split "$AR_SPLIT" --n-clips "$N_AR" \
      --num-frames 16 --size 128 --qps "$QPS" \
      --motion-quantiles "$AR_MOTION_QUANTILES" \
      --spatial-sigma "$AR_MOTION_SIGMA" --post-min-qp "$AR_POST_MIN_QP" \
      --motion-dilation "$AR_MOTION_DILATION" \
      --motion-feather "$AR_MOTION_FEATHER" \
      --ar-backbone "$AR_BACKBONE" --out "$OUT/ar_motion_post" \
      2>&1 | tee "$OUT/ar_motion_post.log"
    echo "[stage] AR motion-preserving POST complete"
  elif [ "$AR_PROBE" = "saliency" ]; then
    echo "[stage] AR action-saliency suppression start"
    python ops/probe_action_saliency.py \
      --index "$INDEX" --split "$AR_SPLIT" --n-clips "$N_AR" \
      --num-frames 16 --size 128 --qps "$QPS" \
      --protect-fractions "$AR_PROTECT_FRACTIONS" \
      --temporal-modes "$AR_SALIENCY_MODES" \
      --sigma "$AR_SALIENCY_SIGMA" \
      --temporal-strength "$AR_TEMPORAL_STRENGTH" \
      --temporal-radius 0 --feather 1 --motion-tau 0.05 \
      --saliency-teacher "$AR_SALIENCY_TEACHER" \
      --eval-backbone "$AR_BACKBONE" --out "$OUT/ar_saliency" \
      2>&1 | tee "$OUT/ar_saliency.log"
    echo "[stage] AR action-saliency suppression complete"
  else
    echo "[stage] AR importance-tube start"
    python ops/probe_action_tubes.py \
      --index "$INDEX" --split "$AR_SPLIT" --n-clips "$N_AR" \
      --num-frames 16 --size 128 --qps "$QPS" \
      --sigmas "$AR_SIGMAS" --temporal-strengths "$TEMPORAL_STRENGTHS" \
      --score 0.2 --dilate 0.30 --feather 8 --ar-backbone "$AR_BACKBONE" \
      --out "$OUT/ar_importance_tubes" 2>&1 | tee "$OUT/ar_importance_tubes.log"
    echo "[stage] AR importance-tube complete"
  fi
fi

if [ "$RUN_OD" = "1" ]; then
  echo "[stage] OD background-suppression start"
  python ops/probe_background_suppression.py \
    --images "$COCO_VAL" --ann "$COCO_ANN" \
    --n-images "$N_OD" --size "$OD_SIZE" --qps "$QPS" \
    --sigmas "$OD_SIGMAS" --score "$OD_SCORE" --dilate "$OD_DILATE" \
    --feather "$OD_FEATHER" --roi-sigmas "$OD_ROI_SIGMAS" \
    --post-sigmas "$OD_POST_SIGMAS" --post-min-qp "$OD_POST_MIN_QP" \
    --min-margin-px "$OD_MIN_MARGIN_PX" --mask-grid "$OD_MASK_GRID" \
    --bootstrap "$BOOTSTRAP" --seed "$OD_SEED" --codecs "$OD_CODECS" \
    --mask-backbone "$OD_MASK_BACKBONE" --eval-backbone "$OD_EVAL_BACKBONE" \
    --out "$OUT/od_background_suppression" 2>&1 | tee "$OUT/od_background_suppression.log"
  echo "[stage] OD background-suppression complete"
fi

echo "[done] selected probes complete (AR=$RUN_AR OD=$RUN_OD)"
