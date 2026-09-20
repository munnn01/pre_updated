import hashlib
from types import SimpleNamespace

from src.engine import _apply_eval_shard


def _canonical_val_samples(n: int) -> list[dict]:
    samples = []
    candidate = 0
    while len(samples) < n:
        key = f"class/video-{candidate}.mp4"
        digest = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        if digest % 10 == 1:
            samples.append({"path": f"/dataset/{key}"})
        candidate += 1
    return samples


def test_salted_eval_shards_do_not_alias_canonical_val_split():
    samples = _canonical_val_samples(100)
    assignments = []
    for shard_idx in range(5):
        ds = SimpleNamespace(samples=list(samples))
        _apply_eval_shard(
            ds,
            {
                "shard_idx": shard_idx,
                "num_shards": 5,
                "shard_salt": "qpc-v4-val-v1",
            },
        )
        assert ds.samples
        assignments.extend(sample["path"] for sample in ds.samples)

    assert len(assignments) == len(samples)
    assert len(set(assignments)) == len(samples)
