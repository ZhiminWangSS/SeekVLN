#!/usr/bin/env python3
"""Standalone SeekVLN+C2PO evaluator.

This is the evaluation entrypoint for the isolated C2PO experiment.  It uses
the post-SFT SeekVLN checkpoint and the dual-mode ``<nav>/<seek>`` parser.  The
implementation is imported from the legacy direct-evaluator module only as a
shared inference core; the AuxThink Base PPO validation dispatcher never
invokes this entrypoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SeekVLN SFT checkpoint with C2PO semantics")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    # Keep --help/config validation usable on login nodes without Habitat
    # installed; the backend is imported only when an actual run is requested.
    from seekvln_auxthink_eval import SeekVLNDirectEvaluator

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"C2PO evaluation config must be a mapping: {config_path}")
    if str(config.get("evaluation_mode", "seekvln_c2po")) != "seekvln_c2po":
        raise ValueError(
            "seekvln_c2po_eval.py requires evaluation_mode=seekvln_c2po; "
            "use the AuxThink validation dispatcher for AuxThink Base checkpoints"
        )
    model_path = str(config.get("model_path", ""))
    if not model_path.strip():
        raise ValueError(
            "SeekVLN evaluation requires a model_path for an exported inference checkpoint; "
            f"got model_path={model_path!r}"
        )
    adapter_path = config.get("lora_adapter_path")
    if adapter_path is not None and not Path(str(adapter_path)).is_dir():
        raise FileNotFoundError(f"C2PO lora_adapter_path does not exist: {adapter_path}")
    evaluator = SeekVLNDirectEvaluator(config)
    try:
        summary = evaluator.run()
    finally:
        evaluator.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
