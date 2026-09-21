from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "ops" / "publish_v12_h264_source.py"
    spec = importlib.util.spec_from_file_location("publish_v12_h264_source", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_dataset_identity_is_account_scoped() -> None:
    module = _module()
    meta = module.dataset_metadata(
        "qktttttttttt", "crc-v5-h264-v12-source-s282001", 282001
    )
    assert meta["id"] == "qktttttttttt/crc-v5-h264-v12-source-s282001"
    assert "private resume source" in meta["title"]


def test_sha256_reads_exact_bytes(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(b"locked-checkpoint")
    assert module.sha256(path) == (
        "9217072ba786c55179ffc32e7afa279f1a18456f7d8e54ae699cbeb4a43c3ca0"
    )
