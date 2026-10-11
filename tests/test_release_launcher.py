import argparse
import copy
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from scripts.eval import run_eval, seekvln_c2po_eval

ROOT = Path(__file__).resolve().parents[1]


def options(**overrides):
    values = dict(model_path=None, output_dir=None, dataset_path=None, gt_path=None,
                  scenes_dir=None, episode_limit=None, episode_start=None, device=None, seed=None)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("dataset", ["r2r", "rxr"])
def test_portable_config_preserves_inference_settings(dataset):
    original = yaml.safe_load((ROOT / "configs" / f"{dataset}.yaml").read_text())
    saved = copy.deepcopy(original)
    config = run_eval.prepare_config(original, options())
    assert original == saved
    for key in ("num_frames", "max_new_tokens", "do_sample", "temperature", "top_p", "top_k",
                "vision_forward_chunk_size", "invalid_action_policy", "max_decisions_per_episode"):
        assert config[key] == original[key]
    assert config["primitive_step_limit_ratio"] is None
    assert "seed" not in config
    assert Path(config["model_path"]).is_absolute()


def test_launcher_accepts_rl_path_and_overrides(tmp_path):
    original = yaml.safe_load((ROOT / "configs/r2r.yaml").read_text())
    model = tmp_path / "SeekVLN-C2PO-RFT"
    config = run_eval.prepare_config(original, options(model_path=str(model), episode_limit=2))
    assert config["model_path"] == str(model)
    assert config["episode_limit"] == 2


def test_preflight_requires_complete_component_layout(tmp_path):
    config = {"model_path": str(tmp_path)}
    with pytest.raises(FileNotFoundError, match="exported model component"):
        run_eval.preflight(config)


@pytest.mark.parametrize("dataset", ["r2r", "rxr"])
def test_preflight_checks_environment_resources(monkeypatch, tmp_path, dataset):
    root = tmp_path / "release"
    model = root / "models/SeekVLN-C2PO-RFT"
    for name in ("config.json", "llm/config.json", "vision_tower/config.json", "mm_projector/config.json"):
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    config = yaml.safe_load((ROOT / "configs" / f"{dataset}.yaml").read_text())
    config["model_path"] = str(model)
    for name in ("dataset_path", "gt_path"):
        path = root / config["habitat"][name].format(split="val_unseen")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        config["habitat"][name] = str(path)
    scenes = root / "scenes"
    scenes.mkdir()
    config["habitat"]["scenes_dir"] = str(scenes)
    import shutil

    shutil.copytree(
        ROOT / "evaluation", root / "evaluation", ignore=shutil.ignore_patterns("data")
    )
    monkeypatch.setattr(run_eval, "ROOT", root)
    if dataset == "r2r":
        with pytest.raises(FileNotFoundError, match="connectivity graphs"):
            run_eval.preflight(config)
        graphs = root / "evaluation/data/connectivity_graphs.pkl"
        graphs.parent.mkdir()
        graphs.write_bytes(b"test")
    run_eval.preflight(config)


def test_c2po_entry_accepts_paper_rl_model_name(monkeypatch, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model_path": "models/SeekVLN-C2PO-RFT"}))
    seen = []

    class Evaluator:
        def __init__(self, config):
            seen.append(config["model_path"])

        def run(self):
            return {"ok": True}

        def close(self):
            seen.append("closed")

    fake = types.ModuleType("seekvln_auxthink_eval")
    fake.SeekVLNDirectEvaluator = Evaluator
    monkeypatch.setitem(sys.modules, "seekvln_auxthink_eval", fake)
    monkeypatch.setattr(sys, "argv", ["seekvln_c2po_eval", "--config", str(path)])
    seekvln_c2po_eval.main()
    assert seen == ["models/SeekVLN-C2PO-RFT", "closed"]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_projector_import_does_not_load_training_wrappers():
    import subprocess

    subprocess.run([sys.executable, "-B", "-c", """
import sys
from llava.trl.models.modeling_auxthink_projector import load_auxthink_base
assert not any(name.startswith(('verl.', 'ray.', 'llava.trl.trainer')) for name in sys.modules)
assert 'llava.trl.models.modeling_value_head' not in sys.modules
"""], check=True)
