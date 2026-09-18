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
    assert cell.index("ops/probe_action_tubes.py") < cell.index(
        "ops/probe_background_suppression.py"
    )
    assert "trap finish EXIT" in cell
    assert "joint_probe_outputs.tgz" in cell
    assert 'RUN_AR="1"' in cell
    assert 'RUN_OD="1"' in cell
    assert "awsaf49/coco-2017-dataset" in cell
    assert "qktttttttttt/kineticscleaned" in cell


def test_joint_notebook_and_metadata_are_kaggle_serializable():
    nb = notebook(render_cell("deadbeef", "confirmatory"))
    meta = metadata("alice", "joint-probe", DEFAULT_DATASETS)

    json.dumps(nb)
    json.dumps(meta)
    assert nb["cells"][0]["source"][0] == "%%bash\n"
    assert nb["cells"][0]["id"] == "joint-od-ar-probe"
    assert meta["id"] == "alice/joint-probe"
    assert meta["dataset_sources"] == list(DEFAULT_DATASETS)
    assert meta["enable_gpu"] is True


def test_joint_cell_can_run_only_od_without_rewriting_template():
    cell = render_cell(
        "abc123",
        run_ar=False,
        run_od=True,
        n_od=200,
        od_sigmas="4",
        od_roi_sigmas="0,1",
        od_post_sigmas="0,1",
        od_post_min_qp=40,
        od_min_margin_px=8,
        od_mask_grid=16,
        od_seed=20260918,
        od_codecs="h264",
        qps="30,35,40,45,50",
        bootstrap=1000,
    )

    assert 'RUN_AR="0"' in cell
    assert 'RUN_OD="1"' in cell
    assert 'N_OD="200"' in cell
    assert 'OD_SIGMAS="4"' in cell
    assert 'OD_ROI_SIGMAS="0,1"' in cell
    assert 'OD_POST_SIGMAS="0,1"' in cell
    assert 'OD_POST_MIN_QP="40"' in cell
    assert 'OD_MIN_MARGIN_PX="8"' in cell
    assert 'OD_MASK_GRID="16"' in cell
    assert 'OD_SEED="20260918"' in cell
    assert 'OD_CODECS="h264"' in cell
    assert '--seed "$OD_SEED"' in cell
    assert '--codecs "$OD_CODECS"' in cell
    assert 'QPS="30,35,40,45,50"' in cell
    assert 'BOOTSTRAP="1000"' in cell
    assert 'if [ "$RUN_AR" = "1" ]; then\n  python scripts/build_train_index.py' in cell
    assert 'ERROR: RUN_OD=1 requires awsaf49/coco-2017-dataset' in cell
    assert 'ERROR: RUN_AR=1 requires qktttttttttt/kineticscleaned' in cell


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_od": 0}, "n_od"),
        ({"od_sigmas": "4;rm"}, "od_sigmas"),
        ({"od_roi_sigmas": "0;rm"}, "od_roi_sigmas"),
        ({"od_post_sigmas": "1;rm"}, "od_post_sigmas"),
        ({"od_post_min_qp": -1}, "od_post_min_qp"),
        ({"od_min_margin_px": -1}, "od_min_margin_px"),
        ({"od_mask_grid": 0}, "od_mask_grid"),
        ({"od_seed": -1}, "od_seed"),
        ({"od_codecs": "h264,hevc"}, "od_codecs"),
        ({"od_codecs": "h264,h264"}, "od_codecs"),
        ({"qps": "30;45"}, "qps"),
        ({"bootstrap": -1}, "bootstrap"),
    ],
)
def test_joint_cell_rejects_unsafe_od_overrides(kwargs, message):
    with pytest.raises(ValueError, match=message):
        render_cell("abc123", **kwargs)


@pytest.mark.parametrize("ref", ["", "bad ref", "\n"])
def test_joint_cell_rejects_unsafe_ref(ref):
    with pytest.raises(ValueError):
        render_cell(ref)


def test_kaggle_command_falls_back_to_cli_entrypoint(monkeypatch):
    monkeypatch.setattr("ops.push_joint_probe.shutil.which", lambda _name: None)

    command = kaggle_command()

    assert command[1] == "-c"
    assert "kaggle.cli" in command[2]
