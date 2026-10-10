"""Portable launcher around the unchanged direct evaluator."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def prepare_config(config: dict, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(config)
    for name in ("model_path", "output_dir"):
        override = getattr(args, name, None)
        config[name] = resolve_path(override or config[name])
    habitat = config["habitat"]
    for name in ("dataset_path", "gt_path", "scenes_dir"):
        override = getattr(args, name, None)
        habitat[name] = resolve_path(override or habitat[name])
    if args.episode_limit is not None:
        if args.episode_limit < 1:
            raise ValueError("episode-limit must be positive")
        config["episode_limit"] = args.episode_limit
    if args.episode_start is not None:
        if args.episode_start < 0:
            raise ValueError("episode-start must be non-negative")
        config["episode_start"] = args.episode_start
    if args.device:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if config.get("primitive_step_limit_ratio") is not None:
        raise ValueError("Public evaluation must not enable the training expert-length cap")
    return config


def preflight(config: dict) -> None:
    model = Path(config["model_path"])
    for name in ("config.json", "llm/config.json", "vision_tower/config.json", "mm_projector/config.json"):
        if not (model / name).is_file():
            raise FileNotFoundError(f"Missing exported model component: {model / name}")
    habitat = config["habitat"]
    for name in ("dataset_path", "gt_path"):
        path = Path(habitat[name].format(split=habitat["split"], role="guide"))
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if not Path(habitat["scenes_dir"]).is_dir():
        raise FileNotFoundError(f"Missing scenes directory: {habitat['scenes_dir']}")
    exp_path = Path(resolve_path(habitat["exp_config"]))
    exp = yaml.safe_load(exp_path.read_text())
    task_path = Path(exp["BASE_TASK_CONFIG_PATH"])
    if not task_path.is_absolute():
        task_path = ROOT / "evaluation" / task_path
    task = yaml.safe_load(task_path.read_text())
    if "TOP_DOWN_MAP_VLNCE" in task.get("TASK", {}).get("MEASUREMENTS", []):
        graph_value = task.get("TASK", {}).get("TOP_DOWN_MAP_VLNCE", {}).get(
            "GRAPHS_FILE", "data/connectivity_graphs.pkl"
        )
        graph = Path(graph_value)
        if not graph.is_absolute():
            graph = ROOT / "evaluation" / graph
        if not graph.is_file():
            raise FileNotFoundError(f"Missing R2R map connectivity graphs: {graph}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an exported SeekVLN SFT or RL model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--gt-path")
    parser.add_argument("--scenes-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--episode-start", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Evaluation config must be a mapping")
    config = prepare_config(raw, args)
    preflight(config)
    if args.check_only:
        print(json.dumps(config, indent=2))
        return
    # Preserve the original RNG behavior unless a seed is explicitly supplied.
    if "seed" in config:
        import random
        import numpy as np
        import torch

        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
    from seekvln_auxthink_eval import SeekVLNDirectEvaluator

    evaluator = SeekVLNDirectEvaluator(config)
    try:
        print(json.dumps(evaluator.run(), indent=2))
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
