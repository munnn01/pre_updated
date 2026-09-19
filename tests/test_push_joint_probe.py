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
    assert "ops/probe_action_motion_post.py" in cell
    assert "ops/probe_action_saliency.py" in cell
    assert "ops/probe_action_guarded.py" in cell
    assert cell.index("ops/probe_action_tubes.py") < cell.index(
        "ops/probe_background_suppression.py"
    )
    assert "trap finish EXIT" in cell
    assert "joint_probe_outputs.tgz" in cell
    assert 'RUN_AR="1"' in cell
    assert 'RUN_OD="1"' in cell
    assert "awsaf49/coco-2017-dataset" in cell
    assert "qktttttttttt/kineticscleaned" in cell
    assert 'OD_POST_MIN_QP="${OD_POST_MIN_QP:-45}"' in cell


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
        n_ar=33,
        od_size=480,
        od_sigmas="4",
        od_roi_sigmas="0,1",
        od_post_sigmas="0,1",
        od_post_min_qp=40,
        od_min_margin_px=8,
        od_mask_grid=16,
        od_seed=20260918,
        od_codecs="h264",
        od_mask_backbone="fasterrcnn_resnet50_fpn",
        od_eval_backbone="fcos_resnet50_fpn",
        ar_probe="post",
        ar_split="val",
        ar_backbone="mc3_18",
        ar_post_sigmas="1,2",
        ar_post_min_qp=45,
        ar_motion_quantiles="0.5,0.75",
        ar_motion_sigma=1.0,
        ar_motion_dilation=3,
        ar_motion_feather=1,
        ar_saliency_teacher="r3d_18",
        ar_protect_fractions="0.15,0.25,0.4",
        ar_saliency_modes="clip,tube",
        ar_saliency_sigma=8,
        ar_temporal_strength=0.75,
        ar_guard_protect_fractions="0.65,0.8",
        ar_guard_motion_fractions="0.5",
        ar_guard_max_blends="0.25,0.4",
        ar_guard_sigma=2,
        ar_guard_retention=0.97,
        ar_guard_blend_steps=4,
        ar_guard_temporal_strength=0.1,
        ar_guard_feather=1,
        qps="30,35,40,45,50",
        bootstrap=1000,
    )

    assert 'RUN_AR="0"' in cell
    assert 'RUN_OD="1"' in cell
    assert 'N_OD="200"' in cell
    assert 'N_AR="33"' in cell
    assert 'OD_SIZE="480"' in cell
    assert 'OD_SIGMAS="4"' in cell
    assert 'OD_ROI_SIGMAS="0,1"' in cell
    assert 'OD_POST_SIGMAS="0,1"' in cell
    assert 'OD_POST_MIN_QP="40"' in cell
    assert 'OD_MIN_MARGIN_PX="8"' in cell
    assert 'OD_MASK_GRID="16"' in cell
    assert 'OD_SEED="20260918"' in cell
    assert 'OD_CODECS="h264"' in cell
    assert 'OD_MASK_BACKBONE="fasterrcnn_resnet50_fpn"' in cell
    assert 'OD_EVAL_BACKBONE="fcos_resnet50_fpn"' in cell
    assert 'AR_PROBE="post"' in cell
    assert 'AR_SPLIT="val"' in cell
    assert 'AR_BACKBONE="mc3_18"' in cell
    assert 'AR_POST_SIGMAS="1,2"' in cell
    assert 'AR_POST_MIN_QP="45"' in cell
    assert 'AR_MOTION_QUANTILES="0.5,0.75"' in cell
    assert 'AR_MOTION_SIGMA="1"' in cell
    assert 'AR_MOTION_DILATION="3"' in cell
    assert 'AR_MOTION_FEATHER="1"' in cell
    assert 'AR_SALIENCY_TEACHER="r3d_18"' in cell
    assert 'AR_PROTECT_FRACTIONS="0.15,0.25,0.4"' in cell
    assert 'AR_SALIENCY_MODES="clip,tube"' in cell
    assert 'AR_SALIENCY_SIGMA="8"' in cell
    assert 'AR_TEMPORAL_STRENGTH="0.75"' in cell
    assert 'AR_GUARD_PROTECT_FRACTIONS="0.65,0.8"' in cell
    assert 'AR_GUARD_MOTION_FRACTIONS="0.5"' in cell
    assert 'AR_GUARD_MAX_BLENDS="0.25,0.4"' in cell
    assert 'AR_GUARD_SIGMA="2"' in cell
    assert 'AR_GUARD_RETENTION="0.97"' in cell
    assert 'AR_GUARD_BLEND_STEPS="4"' in cell
    assert 'AR_GUARD_TEMPORAL_STRENGTH="0.1"' in cell
    assert 'AR_GUARD_FEATHER="1"' in cell
    assert '--seed "$OD_SEED"' in cell
    assert '--codecs "$OD_CODECS"' in cell
    assert '--mask-backbone "$OD_MASK_BACKBONE"' in cell
    assert '--eval-backbone "$OD_EVAL_BACKBONE"' in cell
    assert "ops/probe_action_post.py" in cell
    assert 'QPS="30,35,40,45,50"' in cell
    assert 'BOOTSTRAP="1000"' in cell
    assert 'if [ "$RUN_AR" = "1" ]; then\n  python scripts/build_train_index.py' in cell
    assert 'ERROR: RUN_OD=1 requires awsaf49/coco-2017-dataset' in cell
    assert 'ERROR: RUN_AR=1 requires qktttttttttt/kineticscleaned' in cell


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_od": 0}, "n_od"),
        ({"n_ar": 0}, "n_ar"),
        ({"od_size": 0}, "od_size"),
        ({"od_sigmas": "4;rm"}, "od_sigmas"),
        ({"od_roi_sigmas": "0;rm"}, "od_roi_sigmas"),
        ({"od_post_sigmas": "1;rm"}, "od_post_sigmas"),
        ({"od_post_min_qp": -1}, "od_post_min_qp"),
        ({"od_min_margin_px": -1}, "od_min_margin_px"),
        ({"od_mask_grid": 0}, "od_mask_grid"),
        ({"od_seed": -1}, "od_seed"),
        ({"od_codecs": "h264,hevc"}, "od_codecs"),
        ({"od_codecs": "h264,h264"}, "od_codecs"),
        ({"od_mask_backbone": "yolo"}, "od_mask_backbone"),
        ({"od_eval_backbone": "yolo"}, "od_eval_backbone"),
        ({"ar_probe": "unknown"}, "ar_probe"),
        ({"ar_split": "holdout"}, "ar_split"),
        ({"ar_backbone": "slowfast"}, "ar_backbone"),
        ({"ar_post_sigmas": "1;rm"}, "ar_post_sigmas"),
        ({"ar_post_min_qp": -1}, "ar_post_min_qp"),
        ({"ar_motion_quantiles": "0,0.5"}, "ar_motion_quantiles"),
        ({"ar_motion_quantiles": "0.5,0.5"}, "ar_motion_quantiles"),
        ({"ar_motion_sigma": 0}, "ar_motion_sigma"),
        ({"ar_motion_dilation": -1}, "ar_motion_dilation"),
        ({"ar_motion_feather": -1}, "ar_motion_feather"),
        ({"ar_saliency_teacher": "slowfast"}, "ar_saliency_teacher"),
        ({"ar_protect_fractions": "0,0.25"}, "ar_protect_fractions"),
        ({"ar_protect_fractions": "0.25,0.25"}, "ar_protect_fractions"),
        ({"ar_saliency_modes": "clip,frame"}, "ar_saliency_modes"),
        ({"ar_saliency_sigma": 0}, "ar_saliency_sigma"),
        ({"ar_temporal_strength": 1.1}, "ar_temporal_strength"),
        ({"ar_guard_protect_fractions": "0,0.65"}, "ar_guard_protect_fractions"),
        ({"ar_guard_motion_fractions": "0.5,0.5"}, "ar_guard_motion_fractions"),
        ({"ar_guard_max_blends": "1"}, "ar_guard_max_blends"),
        ({"ar_guard_sigma": 0}, "ar_guard_sigma"),
        ({"ar_guard_retention": 0}, "ar_guard_retention"),
        ({"ar_guard_blend_steps": 0}, "ar_guard_blend_steps"),
        ({"ar_guard_temporal_strength": 1.1}, "ar_guard_temporal_strength"),
        ({"ar_guard_feather": -1}, "ar_guard_feather"),
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
