"""Tests for the COCO detection data path (src/data/coco_det.py).

The index builder is where a one-key mistake already cost a 12 h run: the val
split came out empty, which silently disabled validation and early stopping. The
empty-split assert and the box-scaling convention are pinned here.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_det import (  # noqa: E402
    CocoDetDataset,
    build_coco_index,
    collate_coco_det,
    letterbox_params,
)

def _index(tmp_path: Path) -> Path:
    """Build a self-contained COCO fixture instead of relying on C:\\tmp."""
    images_dir = tmp_path / "val2017"
    ann_dir = tmp_path / "annotations"
    images_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)
    images, annotations = [], []
    for i in range(12):
        name = f"{i:012d}.jpg"
        Image.new("RGB", (40 + i, 30 + i), color=(20 + i, 40, 60)).save(images_dir / name)
        images.append({"id": i + 1, "file_name": name, "width": 40 + i,
                       "height": 30 + i})
        annotations.append({"id": i + 1, "image_id": i + 1, "category_id": 1,
                            "bbox": [4, 5, 12, 10], "area": 120, "iscrowd": 0})
    ann_file = ann_dir / "instances_val2017.json"
    ann_file.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "thing"}],
    }), encoding="utf-8")
    return build_coco_index(images_dir, ann_file, tmp_path / "idx.json",
                            n_train=6, n_val=4)


def test_index_splits_are_non_empty(tmp_path):
    """A non-empty val split is what keeps model selection alive."""
    idx = _index(tmp_path)
    d = json.loads(Path(idx).read_text())
    assert len(d["train"]) == 6 and len(d["val"]) == 4
    assert all(d[k] for k in ("train", "val")), "a split came out empty"


def test_item_shapes_and_box_scaling(tmp_path):
    ds = CocoDetDataset(str(_index(tmp_path)), split="train", frame_size=320)
    clip, tgt = ds[0]
    assert clip.shape == (3, 1, 320, 320)
    assert clip.min() >= 0.0 and clip.max() <= 1.0
    assert tgt["boxes"].shape[1] == 4 and tgt["labels"].ndim == 1
    # boxes must live inside the resized frame
    if len(tgt["boxes"]):
        assert tgt["boxes"].min() >= 0.0
        assert tgt["boxes"].max() <= 320.0 + 1e-3


def test_collate_returns_list_of_targets(tmp_path):
    ds = CocoDetDataset(str(_index(tmp_path)), split="train", frame_size=64)
    clips, targets = collate_coco_det([ds[0], ds[1]])
    assert clips.shape == (2, 3, 1, 64, 64)
    assert isinstance(targets, list) and len(targets) == 2
    assert set(targets[0]) == {"boxes", "labels"}


def test_letterbox_preserves_aspect_ratio():
    scale, left, top, width, height = letterbox_params(400, 200, 320)
    assert scale == 0.8
    assert (width, height) == (320, 160)
    assert left == 0 and top == 80
