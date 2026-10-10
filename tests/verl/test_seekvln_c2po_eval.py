from pathlib import Path

import pytest
import yaml

import torch

import seekvln_auxthink_eval as eval_module
from seekvln_auxthink_eval import (
    _compatible_lora_config,
    _configure_vision_forward_chunking,
    _last_token_logits,
)


ROOT = Path(__file__).resolve().parents[2]


def test_c2po_eval_config_selects_standalone_sft_entrypoint():
    path = ROOT / "configs/r2r.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["evaluation_mode"] == "seekvln_c2po"
    assert config["model_path"] == "models/SeekVLN-FRG-SFT"
    assert config["primitive_step_limit_ratio"] is None


def test_direct_eval_applies_verl_vision_chunking_contract():
    class Model:
        def __init__(self):
            self.tower = type("Tower", (), {})()

        def get_vision_tower(self):
            return self.tower

    model = Model()
    assert _configure_vision_forward_chunking(model, 4) == 4
    assert model.tower.seekvln_forward_chunk_size == 4
    with pytest.raises(ValueError, match="non-negative"):
        _configure_vision_forward_chunking(model, -1)


def test_mode_scoring_materializes_only_the_last_vocabulary_row():
    class Backbone:
        def __call__(self, **kwargs):
            del kwargs
            return type("Output", (), {"last_hidden_state": torch.arange(24.0).reshape(1, 3, 8)})()

    class Head:
        def __call__(self, hidden):
            assert hidden.shape == (1, 1, 8)
            return torch.stack((hidden[..., 0], hidden[..., 1]), dim=-1)

    llm = type("LLM", (), {"model": Backbone(), "lm_head": Head()})()
    logits = _last_token_logits(llm, use_cache=False, return_dict=True)
    assert logits.shape == (1, 2)
    assert logits.tolist() == [[16.0, 17.0]]


def test_mode_scoring_diagnostics_use_the_last_token_logits(monkeypatch):
    evaluator = object.__new__(eval_module.SeekVLNDirectEvaluator)
    evaluator.control_ids = {"<nav>": 1, "<seek>": 2}
    evaluator.model = type("Model", (), {"get_llm": lambda self: object()})()
    evaluator._prepare_multimodal_inputs = lambda prompt_ids, frames: (
        torch.tensor([[1]]),
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.zeros(1, 1, 2),
        [1, 1],
    )
    monkeypatch.setattr(
        eval_module,
        "_last_token_logits",
        lambda llm, **kwargs: torch.tensor([[0.0, 3.0, 2.0]], dtype=torch.float16),
    )

    mode, scores, image_shape, diagnostics = evaluator._score_mode([1], [])

    assert mode == "nav"
    assert scores == [3.0, 2.0]
    assert image_shape == [1, 1]
    assert diagnostics["mode_score_dtype"] == "torch.float16"
    assert diagnostics["mode_score_margin"] == 1.0


def test_eval_launcher_enables_expandable_cuda_segments():
    launcher = (ROOT / "scripts/eval/seekvln_eval.sh").read_text(encoding="utf-8")
    assert "PYTORCH_CUDA_ALLOC_CONF" in launcher
    assert "expandable_segments:True" in launcher


def test_new_adapter_config_is_filtered_for_the_pinned_eval_peft(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type":"LORA","task_type":"CAUSAL_LM","r":8,'
        '"lora_alpha":16,"target_modules":["q_proj"],"exclude_modules":null,'
        '"future_field":123}',
        encoding="utf-8",
    )

    config, ignored = _compatible_lora_config(tmp_path)

    assert config.r == 8
    assert config.lora_alpha == 16
    assert "future_field" in ignored
    # PEFT 0.9 rejects this key while newer training PEFT accepts it. The
    # helper follows the actual evaluator constructor in either environment.
    import inspect
    from peft import LoraConfig

    assert ("exclude_modules" in ignored) == (
        "exclude_modules" not in inspect.signature(LoraConfig).parameters
    )
