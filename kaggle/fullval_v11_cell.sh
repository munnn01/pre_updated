set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
CODEC="__CODEC__"
EXPECTED_SHA="__EXPECTED_SHA__"
REPO="/tmp/pre_updated_fullval_v11"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/fullval_v11/${CODEC}"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "fullval_v11_${CODEC}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/fullval_v11_${CODEC}.tgz"
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

case "$CODEC" in
  h264)
    SOURCE="$(find /kaggle/input -type f \
      -path '*/outputs/dual_checkpoint_v10/h264/t0_lr15/checkpoints/preprocessor.pth' \
      -print -quit || true)"
    CONFIG="configs/crc_v5_per_codec_ar.yaml"
    ;;
  h265)
    SOURCE="$(find /kaggle/input -type f \
      -path '*crc-v5-h265-minus24-candidate-v1*' \
      -name 'preprocessor.pth' -print -quit || true)"
    CONFIG="configs/crc_v5_ar.yaml"
    ;;
  *)
    echo "ERROR: unsupported codec=$CODEC" >&2
    exit 3
    ;;
esac

if [ -z "$SOURCE" ]; then
  echo "ERROR: source checkpoint for $CODEC was not found" >&2
  find /kaggle/input -type f -name 'preprocessor.pth' -print || true
  exit 4
fi
ACTUAL_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "ERROR: checkpoint SHA mismatch expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 5
fi
echo "[source] codec=$CODEC checkpoint=$SOURCE sha256=$ACTUAL_SHA"

python - "$SOURCE" "$CODEC" <<'PY'
import math
import sys

import torch

path, codec = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
assert state["global_step"] == 500, state["global_step"]
cfg = state["cfg"]
assert cfg["model"]["arch"] == "additive_cond"
assert int(cfg["model"]["cond_dim"]) == 3
if codec == "h264":
    assert cfg["codec"]["ste_codec"] == "h264"
    assert cfg["codec"]["ste_alternate"] is False
    rc = cfg["loss"]["rate_constraint"]["per_codec"]["h264"]
    assert math.isclose(float(rc["target_ratio"]), 0.0, abs_tol=1e-12), rc
    assert math.isclose(float(rc["dual_lr"]), 0.015, abs_tol=1e-12), rc
print(f"[checkpoint] codec={codec} step=500 arch=additive_cond cond_dim=3")
PY

mkdir -p "$ROOT_OUT"
python evaluate.py --config "$CONFIG" \
  --ckpt "$SOURCE" --out "$ROOT_OUT" \
  data.index="$INDEX" eval.split=val eval.codecs="[$CODEC]" \
  eval.shard_idx=0 eval.num_shards=1 eval.shard_salt=fullval-v11-locked \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$ROOT_OUT/eval.log"

python - "$ROOT_OUT/results.json" "$ROOT_OUT/manifest.json" \
  "$CODEC" "$EXPECTED_SHA" "$REF" <<'PY'
import json
import math
import sys

result_path, manifest_path, codec, source_sha, commit = sys.argv[1:]
with open(result_path, encoding="utf-8") as handle:
    report = json.load(handle)
assert report["n_eval"] == 1010, report["n_eval"]
assert report["task"] == "action_recognition"
assert report["metric"] == "top1"
metric = report["bd_prep_gain"][f"prep+{codec} vs {codec}"]
assert math.isfinite(float(metric["bd_rate_pct"])), metric
assert math.isfinite(float(metric["bd_accuracy"])), metric
assert report["curves"][codec]["keys"] == [50, 45, 40, 35, 30]
assert report["curves"][f"prep+{codec}"]["keys"] == [50, 45, 40, 35, 30]
manifest = {
    "protocol": "fullval-v11",
    "commit": commit,
    "codec": codec,
    "source_sha256": source_sha,
    "n_eval": report["n_eval"],
    "bd_rate_pct": float(metric["bd_rate_pct"]),
    "bd_accuracy": float(metric["bd_accuracy"]),
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
print("[fullval]", json.dumps(manifest, sort_keys=True))
PY

test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$EXPECTED_SHA"
echo "[done] full-validation V11 codec=$CODEC n_eval=1010"
