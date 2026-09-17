import json

import pytest

from ops.push_joint_probe import (
    DEFAULT_DATASETS,
    kaggle_command,
    metadata,
    notebook,
    render_cell,
)


def test_joint_cell_pins_ref_and_runs_both_probes():
    cell = render_cell("abc123", "quick")

    assert cell.startswith("%%bash\n")
    assert 'REF="abc123"' in cell
    assert 'PROFILE="quick"' in cell
    assert "ops/probe_background_suppression.py" in cell
    assert "ops/probe_action_tubes.py" in cell
    assert "awsaf49/coco-2017-dataset" in cell
    assert "qktttttttttt/kineticscleaned" in cell


def test_joint_notebook_and_metadata_are_kaggle_serializable():
    nb = notebook(render_cell("deadbeef", "confirmatory"))
    meta = metadata("alice", "joint-probe", DEFAULT_DATASETS)

    json.dumps(nb)
    json.dumps(meta)
    assert nb["cells"][0]["source"][0] == "%%bash\n"
    assert meta["id"] == "alice/joint-probe"
    assert meta["dataset_sources"] == list(DEFAULT_DATASETS)
    assert meta["enable_gpu"] is True


@pytest.mark.parametrize("ref", ["", "bad ref", "\n"])
def test_joint_cell_rejects_unsafe_ref(ref):
    with pytest.raises(ValueError):
        render_cell(ref)


def test_kaggle_command_falls_back_to_cli_entrypoint(monkeypatch):
    monkeypatch.setattr("ops.push_joint_probe.shutil.which", lambda _name: None)

    command = kaggle_command()

    assert command[1] == "-c"
    assert "kaggle.cli" in command[2]
