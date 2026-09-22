set -euo pipefail
export PYTHONUNBUFFERED=1

REF="__REF__"
CODEC="__CODEC__"
SEED="__SEED__"
EXPECTED_SOURCE_SHA="__EXPECTED_SOURCE_SHA__"
REPO="/tmp/pre_updated_v13_task_regret"
INDEX="/kaggle/working/kinetics_hash_split.json"
ROOT_OUT="/kaggle/working/outputs/v13_task_regret_screen/${CODEC}"
TRAIN_OUT="$ROOT_OUT/train"
EVAL_OUT="$ROOT_OUT/eval_screen208"

finish() {
  rc=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf "v13_task_regret_screen_${CODEC}.tgz" outputs kinetics_hash_split.json 2>/dev/null
  rm -rf "$REPO"
  echo "[exit] rc=$rc artifact=/kaggle/working/v13_task_regret_screen_${CODEC}.tgz"
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
      -path '*crc-v5-h264-v12-source-s282001*' \
      -name 'preprocessor_last.pth' -print -quit || true)"
    ;;
  h265)
    SOURCE="$(find /kaggle/input -type f \
      -path '*crc-v5-h265-minus24-candidate-v1*' \
      -name 'preprocessor.pth' -print -quit || true)"
    ;;
  *)
    echo "ERROR: unsupported codec=$CODEC" >&2
    exit 3
    ;;
esac
if [ -z "$SOURCE" ]; then
  echo "ERROR: V13 source checkpoint for $CODEC was not found" >&2
  find /kaggle/input -type f -name 'preprocessor*.pth' -print || true
  exit 4
fi
SOURCE_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "ERROR: source SHA mismatch expected=$EXPECTED_SOURCE_SHA actual=$SOURCE_SHA" >&2
  exit 5
fi

python - "$SOURCE" "$CODEC" <<'PY'
import sys
import torch

path, codec = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
assert state["epoch"] == 1, state["epoch"]
assert state["global_step"] == 500, state["global_step"]
assert state.get("opt"), "source has no optimizer audit state"
cfg = state["cfg"]
assert cfg["model"]["arch"] == "additive_cond"
assert int(cfg["model"]["cond_dim"]) == 3
if codec == "h264":
    assert cfg["codec"]["ste_codec"] == "h264"
    assert cfg["codec"]["ste_alternate"] is False
print(f"[source] codec={codec} epoch=1 step=500 path={path}")
PY

mkdir -p "$TRAIN_OUT/checkpoints" "$EVAL_OUT"
# Use the last 500-step model for both codecs.  V13 intentionally starts a
# fresh optimizer and fresh task-dual state because the constrained direction
# changes; it does not inherit CRC-V5's rate duals.
cp "$SOURCE" "$TRAIN_OUT/checkpoints/preprocessor.pth"

python train.py --config configs/v13_task_regret_ar.yaml \
  data.index="$INDEX" out_dir="$TRAIN_OUT" seed="$SEED" \
  codec.ste_codec="$CODEC" codec.ste_alternate=false \
  train.epochs=3 train.max_steps=null \
  train.resume=false train.finetune=true train.cosine=false train.patience=0 \
  loss.rate_constraint.enabled=false \
  loss.task_regret_constraint.enabled=true \
  2>&1 | tee "$TRAIN_OUT/train.log"

FINAL="$TRAIN_OUT/checkpoints/preprocessor_last.pth"
if [ ! -f "$FINAL" ]; then
  echo "ERROR: V13 did not produce $FINAL" >&2
  exit 6
fi
FINAL_SHA="$(sha256sum "$FINAL" | awk '{print $1}')"
python - "$FINAL" "$CODEC" <<'PY'
import math
import sys
import torch

path, codec = sys.argv[1:]
state = torch.load(path, map_location="cpu", weights_only=False)
assert state["epoch"] == 3, state["epoch"]
assert state["global_step"] == 12948, state["global_step"]
cfg = state["cfg"]
assert cfg["codec"]["ste_codec"] == codec
assert cfg["codec"]["ste_alternate"] is False
assert cfg["train"]["finetune"] is True
assert cfg["train"]["resume"] is False
assert cfg["loss"]["rate_constraint"]["enabled"] is False
tc = cfg["loss"]["task_regret_constraint"]
assert tc["enabled"] is True
assert float(tc["epsilon"]) == 0.0
assert float(tc["dual_lr"]) == 0.001
assert float(tc["mu_init"]) == 0.1
assert float(tc["mu_max"]) == 10.0
assert float(tc["rate_weight"]) == 1.0
duals = state["task_duals"]
assert set(duals) == {f"{codec}:{qp}" for qp in (30, 35, 40, 45, 50)}, duals
assert all(0.0 <= float(value) <= 10.0 and math.isfinite(float(value)) for value in duals.values())
print(f"[v13-checkpoint] codec={codec} epoch=3 step=12948 task_duals={duals}")
PY

python evaluate.py --config configs/v13_task_regret_ar.yaml \
  --ckpt "$FINAL" --out "$EVAL_OUT" \
  data.index="$INDEX" eval.split=val eval.codecs="[$CODEC]" \
  eval.shard_idx=0 eval.num_shards=5 \
  eval.shard_salt=v13-task-regret-screen-v1 \
  eval.per_sequence=true eval.include_proxy=false \
  eval.held_out_backbone=r2plus1d_18 \
  2>&1 | tee "$EVAL_OUT/eval.log"

python - "$FINAL" "$EVAL_OUT/results.json" "$ROOT_OUT/manifest.json" \
  "$CODEC" "$EXPECTED_SOURCE_SHA" "$FINAL_SHA" "$SEED" "$REF" <<'PY'
import json
import math
import sys
import torch

(
    checkpoint_path,
    result_path,
    manifest_path,
    codec,
    source_sha,
    final_sha,
    seed,
    commit,
) = sys.argv[1:]
state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
with open(result_path, encoding="utf-8") as handle:
    report = json.load(handle)
assert report["n_eval"] == 208, report["n_eval"]
metric = report["bd_prep_gain"][f"prep+{codec} vs {codec}"]
assert math.isfinite(float(metric["bd_rate_pct"])), metric
assert math.isfinite(float(metric["bd_accuracy"])), metric
assert report["curves"][codec]["keys"] == [50, 45, 40, 35, 30]
assert report["curves"][f"prep+{codec}"]["keys"] == [50, 45, 40, 35, 30]
manifest = {
    "protocol": "v13-task-regret-screen",
    "commit": commit,
    "codec": codec,
    "seed": int(seed),
    "source_sha256": source_sha,
    "final_sha256": final_sha,
    "epoch": int(state["epoch"]),
    "global_step": int(state["global_step"]),
    "n_eval": int(report["n_eval"]),
    "shard_salt": "v13-task-regret-screen-v1",
    "shard_fingerprint": "f321b810af2c507c",
    "bd_rate_pct": float(metric["bd_rate_pct"]),
    "bd_accuracy": float(metric["bd_accuracy"]),
    "task_duals": {
        key: float(value) for key, value in state["task_duals"].items()
    },
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
print("[v13-result]", json.dumps(manifest, sort_keys=True))
PY

test "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$EXPECTED_SOURCE_SHA"
echo "[done] V13 task-regret screen codec=$CODEC n_eval=208"
