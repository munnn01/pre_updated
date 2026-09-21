set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
SEED="__SEED__"
EXPECTED_SHA="__EXPECTED_SHA__"
ARM_ORDER="__ARM_ORDER__"
REPO="/tmp/pre_updated_crc_v5_pc1"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/crc_v5_pc1"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf crc_v5_pc1_paired.tgz outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/crc_v5_pc1_paired.tgz"
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
  echo "ERROR: account-local Stage-1 dataset lacks preprocessor.pth" >&2
  exit 3
fi
ACTUAL_SHA="$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "ERROR: Stage-1 SHA mismatch expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 4
fi
echo "[source] checkpoint=$STAGE1_CKPT sha256=$ACTUAL_SHA order=$ARM_ORDER seed=$SEED"

run_arm() {
  local arm="$1"
  local h264_dual_lr="$2"
  local out="$ROOT_OUT/$arm"
  local eval_out="$out/eval_val1of20"

  mkdir -p "$out/checkpoints" "$eval_out"
  cp "$STAGE1_CKPT" "$out/checkpoints/preprocessor.pth"
  echo "[train] arm=$arm h264_dual_lr=$h264_dual_lr h265_dual_lr=0.005"
  python train.py --config configs/crc_v5_per_codec_ar.yaml \
    data.index="$INDEX" out_dir="$out" seed="$SEED" \
    loss.rate_constraint.per_codec.h264.target_ratio=0.0 \
    loss.rate_constraint.per_codec.h264.dual_lr="$h264_dual_lr" \
    loss.rate_constraint.per_codec.h265.target_ratio=0.0 \
    loss.rate_constraint.per_codec.h265.dual_lr=0.005 \
    2>&1 | tee "$out/train.log"

  local ckpt="$out/checkpoints/preprocessor.pth"
  python - "$ckpt" "$arm" "$h264_dual_lr" "$ACTUAL_SHA" <<'PY'
import json
import math
import sys
import torch

path, arm, expected_h264_lr, source_sha = sys.argv[1:]
state = torch.load(path, map_location="cpu")
cfg = state["cfg"]
rc = cfg["loss"]["rate_constraint"]
assert cfg["codec"]["kind"] == "ste"
assert cfg["codec"]["ste_alternate"] is True
assert cfg["train"]["balanced_codec_qp"] is True
assert state["global_step"] == 500, state["global_step"]
assert float(rc["per_codec"]["h264"]["target_ratio"]) == 0.0
assert float(rc["per_codec"]["h265"]["target_ratio"]) == 0.0
assert float(rc["per_codec"]["h264"]["dual_lr"]) == float(expected_h264_lr)
assert float(rc["per_codec"]["h265"]["dual_lr"]) == 0.005
duals = state.get("rate_duals", {})
assert set(duals) == {
    f"{codec}:{qp}"
    for codec in ("h264", "h265")
    for qp in (30, 35, 40, 45, 50)
}
assert all(math.isfinite(float(value)) for value in duals.values())
print("[checkpoint]", arm, "step=", state["global_step"],
      "source_sha=", source_sha, "duals=", json.dumps(duals, sort_keys=True))
PY

  echo "[eval] arm=$arm fresh validation shard=1/20"
  python evaluate.py --config configs/crc_v5_per_codec_ar.yaml \
    --ckpt "$ckpt" --out "$eval_out" \
    data.index="$INDEX" eval.split=val \
    eval.shard_idx=1 eval.num_shards=20 eval.shard_salt=crc-v5-pc1-val-v1 \
    eval.per_sequence=true eval.include_proxy=false \
    eval.held_out_backbone=r2plus1d_18 \
    2>&1 | tee "$eval_out.log"
  echo "[done-arm] $arm result=$eval_out/results.json"
}

case "$ARM_ORDER" in
  control-first)
    run_arm control 0.005
    run_arm treatment 0.015
    ;;
  treatment-first)
    run_arm treatment 0.015
    run_arm control 0.005
    ;;
  *)
    echo "ERROR: unsupported arm order: $ARM_ORDER" >&2
    exit 5
    ;;
esac

echo "[done] CRC-V5-PC1 paired control/treatment complete"
