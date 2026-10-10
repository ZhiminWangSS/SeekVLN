#!/usr/bin/env python3
"""Direct AuxThink-style SeekVLN evaluation for SFT checkpoints."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import yaml
from PIL import Image
from PIL import ImageDraw
from transformers import GenerationConfig

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model
from llava.trl.models.modeling_auxthink_projector import load_auxthink_base
from llava.utils.tokenizer import infer_stop_tokens
from llava.verl.auxthink_habitat import AuxThinkVerlHabitatBackend
from llava.verl.seekvln_inference import (
    SEEK_ENVIRONMENT_TURN,
    build_seekvln_stage1_prompt,
    parse_seekvln_action,
    parse_seekvln_response,
    tokenize_seekvln_conversation,
)
from llava.verl.seekvln_metrics import aggregate_seekvln_decision_metrics
from llava.verl.seekvln_tokens import ensure_seekvln_tokenizer, initialize_seekvln_control_embeddings


def _optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "false", "disabled", "off"}:
        return None
    return float(value)


def _optional_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "false", "disabled", "off"}:
        return None
    return int(value)


def _sample_auxthink_history(frames: list[Image.Image], max_images: int) -> list[Image.Image]:
    if not frames:
        raise ValueError("SeekVLN history sampler received no frames")
    if int(max_images) < 1:
        raise ValueError("max_images must be positive")
    if len(frames) <= int(max_images):
        return list(frames)
    step = (len(frames) - 1) / float(int(max_images) - 1)
    return [frames[int(round(j * step))] for j in range(int(max_images))]


def _auxthink_history_indices(frame_count: int, max_images: int) -> list[int]:
    if int(frame_count) < 1:
        raise ValueError("frame_count must be positive")
    if int(frame_count) <= int(max_images):
        return list(range(int(frame_count)))
    step = (int(frame_count) - 1) / float(int(max_images) - 1)
    return [int(round(j * step)) for j in range(int(max_images))]


def _clean_response(tokenizer, token_ids: list[int]) -> tuple[list[int], str]:
    removable = {
        int(item)
        for item in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if item is not None
    }
    start, end = 0, len(token_ids)
    while start < end and token_ids[start] in removable:
        start += 1
    while end > start and token_ids[end - 1] in removable:
        end -= 1
    cleaned = list(token_ids[start:end])
    return cleaned, tokenizer.decode(cleaned, skip_special_tokens=False).strip()


def _generated_token_ids(output_ids: torch.Tensor, prompt_ids: torch.Tensor) -> list[int]:
    """Extract generated ids from either HF full-sequence or custom generated-only output."""
    if output_ids.ndim != 2 or output_ids.shape[0] < 1:
        raise ValueError(f"Expected a [batch, sequence] tensor, got shape={tuple(output_ids.shape)}")
    sequence = output_ids[0]
    prompt = prompt_ids[0]
    prompt_length = int(prompt.shape[0])
    if sequence.shape[0] >= prompt_length and torch.equal(sequence[:prompt_length], prompt):
        sequence = sequence[prompt_length:]
    return [int(token) for token in sequence.tolist()]


def _configure_vision_forward_chunking(model, configured: int | None) -> int:
    """Apply the same ordered frozen-vision chunking used by VERL rollout."""

    chunk_size = int(4 if configured is None else configured)
    if chunk_size < 0:
        raise ValueError("vision_forward_chunk_size must be non-negative")
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        vision_tower.seekvln_forward_chunk_size = chunk_size
    return chunk_size


def _last_token_logits(llm, **forward_kwargs) -> torch.Tensor:
    """Compute only the final vocabulary row instead of ``[seq, vocab]``."""

    backbone = getattr(llm, "model", None)
    lm_head = getattr(llm, "lm_head", None)
    if backbone is None or lm_head is None:
        raise TypeError("SeekVLN direct evaluation requires a causal LM backbone and lm_head")
    output = backbone(**forward_kwargs)
    hidden = output.last_hidden_state[:, -1:, :]
    return lm_head(hidden)[:, -1, :]


def _compatible_lora_config(adapter_path: Path):
    """Load a new-export adapter with the evaluator's older PEFT version."""

    from peft import LoraConfig

    raw = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    accepted = set(inspect.signature(LoraConfig).parameters)
    filtered = {key: value for key, value in raw.items() if key in accepted}
    ignored = sorted(set(raw) - set(filtered))
    return LoraConfig(**filtered), ignored


def _slim_info(info: dict, max_value_bytes: int = 4096) -> dict:
    """Drop bulky measure payloads (e.g. TOP_DOWN_MAP grids) from the
    per-decision habitat info snapshot; keep scalar/small diagnostics only.
    R2R's TOP_DOWN_MAP_VLNCE serializes to 6-12MB per decision, which turned
    decisions.jsonl into tens of GB per shard and threatened the shared NAS."""
    slim = {}
    for key, value in info.items():
        try:
            encoded = json.dumps(value, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
        except (TypeError, ValueError):
            continue
        if len(encoded) <= max_value_bytes:
            slim[key] = value
    return slim


class SeekVLNDirectEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = str(config["model_path"])
        self.num_frames = int(config.get("num_frames", 9))
        self.max_new_tokens = config.get("max_new_tokens", None)
        self.episode_start = int(config.get("episode_start", 0))
        configured_episode_ids = config.get("episode_ids")
        if configured_episode_ids is not None and not isinstance(configured_episode_ids, list):
            raise ValueError("episode_ids must be a YAML list when provided")
        self.episode_ids = (
            None if not configured_episode_ids else [str(value) for value in configured_episode_ids]
        )
        if self.episode_ids and self.episode_start:
            raise ValueError("episode_start and episode_ids cannot be used together")
        self.do_sample = bool(config.get("do_sample", False))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = float(config.get("top_p", 1.0))
        self.top_k = int(config.get("top_k", -1))
        self.image_save_dir = Path(str(config["image_save_dir"])).expanduser() if config.get("image_save_dir") else None
        self.trajectory_video = bool(config.get("trajectory_video", False))
        self.seek_panorama_dir = Path(str(config["seek_panorama_dir"])).expanduser() if config.get("seek_panorama_dir") else None
        self.trajectory_points = []
        if self.image_save_dir:
            self.image_save_dir.mkdir(parents=True, exist_ok=True)
        self.invalid_action_policy = str(config.get("invalid_action_policy", "auxthink_random"))
        if self.invalid_action_policy != "auxthink_random":
            raise ValueError("SeekVLN direct evaluator currently supports invalid_action_policy=auxthink_random only")
        self.device = torch.device(config.get("device", "cuda:0"))
        self.empty_cuda_cache_each_decision = bool(config.get("empty_cuda_cache_each_decision", True))
        tokenizer, model, image_processor, _ = load_auxthink_base(
            load_pretrained_model,
            self.model_path,
            get_model_name_from_path(self.model_path),
        )
        # C2PO/PPO checkpoints are intentionally stored as portable PEFT
        # adapters so that a two-GPU evaluator does not need to restore the
        # training FSDP world size.  Keep the SFT model as the immutable base
        # and apply the adapter only when the config explicitly requests it.
        adapter_path = config.get("lora_adapter_path")
        self.adapter_ignored_config_keys: list[str] = []
        if adapter_path:
            adapter_path = Path(str(adapter_path)).expanduser().resolve()
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"lora_adapter_path is missing adapter_config.json: {adapter_path}"
                )
            from peft import PeftModel

            adapter_config, ignored_keys = _compatible_lora_config(adapter_path)
            self.adapter_ignored_config_keys = ignored_keys
            model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
                config=adapter_config,
            )
            if hasattr(model, "merge_and_unload") and bool(config.get("merge_lora_adapter", True)):
                model = model.merge_and_unload()
        token_ids = ensure_seekvln_tokenizer(tokenizer)
        self._prepare_seekvln_embeddings(model, tokenizer, token_ids)
        # Match the VERL rollout path: the frozen SigLIP tower processes the
        # same ordered 8-history + 1-current frames in bounded chunks. Selected
        # image features are concatenated in their original order.
        vision_chunk_size = _configure_vision_forward_chunking(
            model, config.get("vision_forward_chunk_size", 4)
        )
        model.to(device=self.device, dtype=torch.float16)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.vision_forward_chunk_size = vision_chunk_size
        self.control_ids = {
            token: int(tokenizer.convert_tokens_to_ids(token))
            for token in ("<nav>", "<seek>", "</seek>", "<think>", "</think>")
        }
        habitat = config["habitat"]
        self.backend = AuxThinkVerlHabitatBackend(
            exp_config=habitat["exp_config"],
            split=habitat["split"],
            dataset_path=habitat["dataset_path"],
            gt_path=habitat["gt_path"],
            scenes_dir=habitat["scenes_dir"],
            # Evaluation must not silently inherit the training 1.3x expert
            # primitive cap.  Set it explicitly only for an ablation.
            primitive_step_limit_ratio=_optional_float(config.get("primitive_step_limit_ratio", None)),
            num_chunks=1,
            chunk_idx=0,
            skip_episodes=self.episode_start,
            episode_ids=self.episode_ids,
        )

    def _prepare_seekvln_embeddings(self, model, tokenizer, token_ids) -> None:
        embedding_rows = int(model.get_llm().get_input_embeddings().weight.shape[0])
        tokenizer_rows = int(len(tokenizer))
        if embedding_rows == tokenizer_rows:
            return
        if embedding_rows > tokenizer_rows:
            raise ValueError(
                f"Model embedding rows exceed tokenizer length: model={embedding_rows}, tokenizer={tokenizer_rows}"
            )
        if not bool(self.config.get("initialize_missing_control_embeddings", False)):
            raise ValueError(
                "Loaded model/tokenizer vocab sizes differ. Refusing to resize and initialize SeekVLN control "
                "embeddings during evaluation because this can overwrite trained SFT token rows. Set "
                "`initialize_missing_control_embeddings: true` only for base-model debugging."
            )
        model.resize_token_embeddings(tokenizer_rows)
        initialize_seekvln_control_embeddings(model, tokenizer, token_ids)

    def close(self):
        self.backend.close()

    def _generate_ids(self, prompt_ids: list[int], images: list[Image.Image]):
        image_count = len(images)
        placeholder_count = sum(int(token) == IMAGE_TOKEN_INDEX for token in prompt_ids)
        if image_count != placeholder_count:
            raise ValueError(f"image/token mismatch: images={image_count}, placeholders={placeholder_count}")
        query = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float16)
        generation_config = self._generation_config()
        with torch.inference_mode():
            output_ids = self.model.generate(
                query,
                images=image_tensor,
                attention_mask=torch.ones_like(query),
                generation_config=generation_config,
                use_cache=True,
            )
        token_ids = _generated_token_ids(output_ids, query)
        generated_token_count = len(token_ids)
        generation_hit_max_new_tokens = (
            self.max_new_tokens is not None and generated_token_count >= int(self.max_new_tokens)
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and token_ids and token_ids[-1] == int(eos_id):
            token_ids.pop()
        return (
            token_ids,
            list(image_tensor.shape),
            generated_token_count,
            bool(generation_hit_max_new_tokens),
        )

    def _generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(getattr(self.model, "generation_config", None) or GenerationConfig())
        if self.max_new_tokens is not None:
            generation_config.max_new_tokens = int(self.max_new_tokens)
        elif generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = int(self.tokenizer.model_max_length)
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.convert_tokens_to_ids(infer_stop_tokens(self.tokenizer))
        generation_config.do_sample = self.do_sample
        generation_config.temperature = self.temperature
        generation_config.top_p = self.top_p
        generation_config.top_k = self.top_k
        return generation_config

    def _prepare_multimodal_inputs(self, prompt_ids: list[int], images: list[Image.Image]):
        image_count = len(images)
        placeholder_count = sum(int(token) == IMAGE_TOKEN_INDEX for token in prompt_ids)
        if image_count != placeholder_count:
            raise ValueError(f"image/token mismatch: images={image_count}, placeholders={placeholder_count}")
        query = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float16)
        input_ids, position_ids, attention_mask, _, inputs_embeds, _ = (
            self.model.prepare_inputs_labels_for_multimodal(
                query,
                None,
                torch.ones_like(query),
                None,
                None,
                image_tensor,
            )
        )
        return input_ids, position_ids, attention_mask, inputs_embeds, list(image_tensor.shape)

    def _score_mode(self, prompt_ids: list[int], frames: list[Image.Image]):
        input_ids, position_ids, attention_mask, inputs_embeds, image_shape = self._prepare_multimodal_inputs(
            prompt_ids,
            frames,
        )
        with torch.inference_mode():
            logits = _last_token_logits(
                self.model.get_llm(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
            )
        mode_score_dtype = str(logits.dtype)
        logits = logits[0].float()
        scores = [float(logits[self.control_ids["<nav>"]]), float(logits[self.control_ids["<seek>"]])]
        mode = "nav" if scores[0] >= scores[1] else "seek"
        diagnostics = {
            "mode_score_source": "raw_inner_llm_forward_last_logit",
            "mode_score_dtype": mode_score_dtype,
            "mode_score_margin": float(scores[0] - scores[1]),
        }
        return mode, scores, image_shape, diagnostics

    def _generate_nav(self, prompt_ids: list[int], frames: list[Image.Image]):
        token_ids, image_shape, generated_token_count, generation_hit_max_new_tokens = self._generate_ids(
            [*prompt_ids, self.control_ids["<nav>"]],
            frames,
        )
        response_ids, text = _clean_response(self.tokenizer, token_ids)
        return (
            response_ids,
            parse_seekvln_response("nav", text),
            image_shape,
            generated_token_count,
            generation_hit_max_new_tokens,
        )

    def _generate_seek(self, stage1_prompt: str, frames: list[Image.Image]):
        panorama = self.backend.get_seekvln_panorama()
        saved_panorama_dir = None
        if self.seek_panorama_dir:
            saved_panorama_dir = self.seek_panorama_dir / f"episode_{self.current_episode_id}" / f"decision_{self.current_decision_index:04d}"
            saved_panorama_dir.mkdir(parents=True, exist_ok=True)
            for name in ("left", "forward", "right", "back"):
                Image.fromarray(np.uint8(panorama[name])).convert("RGB").save(saved_panorama_dir / f"{name}.png")
            self.last_saved_panorama_dir = str(saved_panorama_dir)
        views = [Image.fromarray(np.uint8(panorama[name])).convert("RGB") for name in ("left", "forward", "right")]
        conversation = [
            {"from": "human", "value": stage1_prompt},
            {"from": "gpt", "value": "<seek>"},
            {"from": "human", "value": SEEK_ENVIRONMENT_TURN},
        ]
        prompt_ids = tokenize_seekvln_conversation(self.tokenizer, conversation, add_generation_prompt=True)
        token_ids, image_shape, generated_token_count, generation_hit_max_new_tokens = self._generate_ids(
            [*prompt_ids],
            [*frames, *views],
        )
        response_ids, text = _clean_response(self.tokenizer, token_ids)
        return (
            prompt_ids,
            response_ids,
            parse_seekvln_response("seek", text),
            image_shape,
            generated_token_count,
            generation_hit_max_new_tokens,
        )

    def _executable_actions(self, parsed):
        if parsed.actions:
            return list(parsed.actions), False, parsed.final_action_text
        if self.invalid_action_policy != "auxthink_random":
            return [], False, ""
        candidate = parsed.final_action_text
        if not candidate and parsed.mode == "seek":
            if "</seek><think>" in parsed.text:
                candidate = parsed.text.split("</seek><think>", 1)[1]
            elif "<think>" in parsed.text:
                candidate = parsed.text.rsplit("<think>", 1)[1]
            if "</think>" in candidate:
                candidate = candidate.split("</think>", 1)[0]
        actions = parse_seekvln_action(candidate or parsed.text)
        return actions, True, str(candidate).strip()

    def run(self):
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / "decisions.jsonl"
        summary_path = output_dir / "summary.json"
        episode_limit = int(self.config.get("episode_limit", len(self.episode_ids or []) or 10))
        if self.episode_ids and episode_limit != len(self.episode_ids):
            raise ValueError(
                f"episode_limit={episode_limit} must match len(episode_ids)={len(self.episode_ids)}"
            )
        max_decisions = _optional_int(self.config.get("max_decisions_per_episode"))
        records = []
        with records_path.open("w", encoding="utf-8") as handle:
            for _ in range(episode_limit):
                observation = self.backend.reset()
                episode_id = self.backend.episode_id
                self.current_episode_id = episode_id
                self.trajectory_points = []
                decision_index = 0
                while max_decisions is None or decision_index < max_decisions:
                    self.current_decision_index = decision_index
                    self.last_saved_panorama_dir = None
                    history = [Image.fromarray(np.uint8(frame)).convert("RGB") for frame in observation["auxthink_rgb_history"]]
                    frames = _sample_auxthink_history(history, self.num_frames)
                    history_indices = _auxthink_history_indices(len(history), self.num_frames)
                    if self.image_save_dir:
                        frame_dir = self.image_save_dir / f"episode_{episode_id}" / f"decision_{decision_index:04d}"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        for frame_idx, frame in zip(history_indices, frames):
                            frame.save(frame_dir / f"history_{frame_idx:04d}.png")
                        if self.trajectory_video:
                            from habitat_extensions.utils import observations_to_image
                            composite = observations_to_image(
                                {"rgb": np.asarray(history[-1])}, self.backend.get_metrics()
                            )
                            Image.fromarray(composite).save(frame_dir / "trajectory_frame.png")
                    stage1_prompt = build_seekvln_stage1_prompt(observation["auxthink_instruction"], len(frames))
                    prompt_ids = tokenize_seekvln_conversation(
                        self.tokenizer,
                        [{"from": "human", "value": stage1_prompt}],
                        add_generation_prompt=True,
                    )
                    mode, mode_scores, mode_image_shape, mode_diagnostics = self._score_mode(prompt_ids, frames)
                    if mode == "nav":
                        (
                            response_ids,
                            parsed,
                            image_shape,
                            generated_token_count,
                            generation_hit_max_new_tokens,
                        ) = self._generate_nav(prompt_ids, frames)
                    else:
                        (
                            prompt_ids,
                            response_ids,
                            parsed,
                            image_shape,
                            generated_token_count,
                            generation_hit_max_new_tokens,
                        ) = self._generate_seek(stage1_prompt, frames)
                    actions, invalid_action_executed, fallback_action_text = self._executable_actions(parsed)
                    executable = bool(actions)
                    if executable:
                        observation, done, info = self.backend.step(actions)
                    else:
                        done, info = False, self.backend.get_metrics()
                    record = {
                        "episode_id": episode_id,
                        "decision_index": decision_index,
                        "mode": parsed.mode,
                        "mode_scores": mode_scores,
                        **mode_diagnostics,
                        "format_valid": parsed.format_valid,
                        "invalid_reason": parsed.invalid_reason,
                        "valid_action": executable,
                        "action_fallback": parsed.action_fallback,
                        "invalid_action_executed": bool(invalid_action_executed),
                        "fallback_action_text": fallback_action_text,
                        "parsed_actions": parsed.actions,
                        "actions": actions,
                        "final_action_text": parsed.final_action_text,
                        "response_text": parsed.text,
                        "generation_token_count": int(generated_token_count),
                        "generation_hit_max_new_tokens": bool(generation_hit_max_new_tokens),
                        "image_count": len(frames) if mode == "nav" else len(frames) + 3,
                        "history_image_count": len(frames),
                        "history_total_frames": len(history),
                        "history_sample_indices": history_indices,
                        "history_current_index": len(history) - 1,
                        "history_last_is_current": bool(history_indices[-1] == len(history) - 1),
                        "image_tensor_shape": image_shape,
                        "mode_image_tensor_shape": mode_image_shape,
                        "primitive_steps": int(info.get("primitive_steps", 0)),
                        "expert_primitive_steps": int(info.get("expert_primitive_steps", 0)),
                        "distance_to_goal": info.get("distance_to_goal"),
                        "success": info.get("success"),
                        "spl": info.get("spl"),
                        "ndtw": info.get("ndtw") or info.get("nDTW"),
                        "saved_image_dir": str(frame_dir) if self.image_save_dir else None,
                        "saved_seek_panorama_dir": self.last_saved_panorama_dir,
                        "oracle_success": info.get("oracle_success"),
                        "path_length": info.get("path_length"),
                        "done": bool(done),
                        "current": _slim_info(info),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n")
                    handle.flush()
                    records.append(record)
                    # The standalone evaluator carries the complete model on
                    # one 48GB device. SEEK generation and the following mode
                    # score have different allocation shapes, so returning
                    # unused blocks at the decision boundary avoids allocator
                    # fragmentation across long episodes. This does not alter
                    # model state, inputs, logits, or generated tokens.
                    if self.device.type == "cuda" and self.empty_cuda_cache_each_decision:
                        torch.cuda.empty_cache()
                    if done:
                        break
                    decision_index += 1
        metrics = aggregate_seekvln_decision_metrics(records)
        summary = {
            "records": len(records),
            "episodes": sorted({item["episode_id"] for item in records}, key=lambda value: int(value)),
            "metrics": metrics,
            "nav_decisions": sum(item["mode"] == "nav" for item in records),
            "seek_decisions": sum(item["mode"] == "seek" for item in records),
            "format_valid_decisions": sum(bool(item["format_valid"]) for item in records),
            "invalid_decisions": sum(not bool(item["format_valid"]) for item in records),
            "generation_max_new_tokens": self.max_new_tokens,
            "vision_forward_chunk_size": self.vision_forward_chunk_size,
            "empty_cuda_cache_each_decision": self.empty_cuda_cache_each_decision,
            "adapter_ignored_config_keys": self.adapter_ignored_config_keys,
            "generation_hit_max_new_tokens": sum(
                bool(item["generation_hit_max_new_tokens"]) for item in records
            ),
            "generation_hit_max_new_tokens_rate": (
                sum(bool(item["generation_hit_max_new_tokens"]) for item in records) / len(records)
                if records
                else 0.0
            ),
            "generation_token_count": {
                "max": max((int(item["generation_token_count"]) for item in records), default=0),
                "p95": (
                    int(np.percentile([int(item["generation_token_count"]) for item in records], 95))
                    if records
                    else 0
                ),
            },
            "records_path": str(records_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    evaluator = SeekVLNDirectEvaluator(config)
    try:
        summary = evaluator.run()
    finally:
        evaluator.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
