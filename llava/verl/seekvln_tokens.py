"""SeekVLN tokenizer contract for AuxThink-compatible VERL training."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from llava.constants import SENTINEL_TOKEN


SEEKVLN_CONTROL_TOKENS: tuple[str, ...] = (
    "<nav>",
    "<seek>",
    "</seek>",
    "<think>",
    "</think>",
)
SEEKVLN_MODE_TOKENS: tuple[str, ...] = ("<nav>", "<seek>")
# ``<nav>`` is structural when it appears inside a seek response as the final
# action marker.  The first response token is still handled separately as mode CE.
SEEKVLN_STRUCT_TOKENS: tuple[str, ...] = ("</seek>", "<think>", "</think>", "<nav>")
AUXTHINK_MEDIA_TOKENS: tuple[str, ...] = ("<image>", "<vila/video>")


@dataclass(frozen=True)
class SeekVLNTokenIds:
    """Resolved single-token ids used by SeekVLN SFT loss and inference."""

    control: dict[str, int]
    mode: list[int]
    response_special: list[int]
    added_control_tokens: list[str]
    sentinel_token_id: int


def normalized_control_tokens(tokens: Iterable[str] | None = None) -> list[str]:
    """Return the canonical 3.3 control-token list, preserving valid overrides."""

    if tokens is None:
        tokens = SEEKVLN_CONTROL_TOKENS
    result: list[str] = []
    for token in tokens:
        token = str(token).strip()
        if token and token not in result:
            result.append(token)
    missing = [token for token in SEEKVLN_CONTROL_TOKENS if token not in result]
    if missing:
        raise ValueError(f"SeekVLN control tokens are missing required 3.3 tokens: {missing}")
    extra = [token for token in result if token not in SEEKVLN_CONTROL_TOKENS]
    if extra:
        raise ValueError(f"SeekVLN 3.3 control tokens must not include legacy tokens: {extra}")
    return result


def _single_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"SeekVLN control token is missing from tokenizer: {token}")
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [token_id]:
        raise ValueError(f"SeekVLN control token must encode as one token: {token!r} -> {encoded}")
    return int(token_id)


def _ensure_auxthink_tokenizer_compatibility(tokenizer: Any) -> int:
    """Preserve AuxThink's base tokenizer behavior while adding VILA's sentinel.

    The released AuxThink checkpoint is loaded from its componentized LLM
    tokenizer.  SeekVLN must not replace existing media/chat tokens, but VILA's
    conversation masking helper needs ``<vila/sentinel>`` to exist before the
    model embedding resize happens.
    """

    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
    missing_media = [
        token
        for token in AUXTHINK_MEDIA_TOKENS
        if tokenizer.convert_tokens_to_ids(token) in {None, getattr(tokenizer, "unk_token_id", None)}
    ]
    if missing_media:
        raise ValueError(
            "SeekVLN Methodology-3.3 must extend the original AuxThink base tokenizer; "
            f"missing AuxThink media tokens: {missing_media}"
        )
    for token in AUXTHINK_MEDIA_TOKENS:
        _single_token_id(tokenizer, token)
    if not hasattr(tokenizer, "sentinel_token"):
        tokenizer.add_tokens([SENTINEL_TOKEN], special_tokens=True)
        tokenizer.sentinel_token = SENTINEL_TOKEN
    tokenizer.sentinel_token_id = _single_token_id(tokenizer, SENTINEL_TOKEN)
    return int(tokenizer.sentinel_token_id)


def ensure_seekvln_tokenizer(tokenizer: Any, tokens: Iterable[str] | None = None) -> SeekVLNTokenIds:
    """Add and validate Methodology-3.3 controls without replacing AuxThink tokens."""

    control_tokens = normalized_control_tokens(tokens)
    sentinel_token_id = _ensure_auxthink_tokenizer_compatibility(tokenizer)
    preexisting_vocab = dict(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else {}
    added_control_tokens = [token for token in control_tokens if token not in preexisting_vocab]
    try:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": control_tokens},
            replace_additional_special_tokens=False,
        )
    except TypeError:
        tokenizer.add_special_tokens({"additional_special_tokens": control_tokens})
    control_ids = {token: _single_token_id(tokenizer, token) for token in control_tokens}
    return SeekVLNTokenIds(
        control=control_ids,
        mode=[control_ids[token] for token in SEEKVLN_MODE_TOKENS],
        response_special=[control_ids[token] for token in SEEKVLN_STRUCT_TOKENS],
        added_control_tokens=added_control_tokens,
        sentinel_token_id=sentinel_token_id,
    )


def sync_auxthink_worker_tokenizer(tokenizer: Any) -> SeekVLNTokenIds:
    """Apply the rollout-server vocabulary contract to an AuxThink worker."""

    return ensure_seekvln_tokenizer(tokenizer)


def auxthink_uses_seekvln_control_tokens() -> bool:
    """Whether this runtime is the dual-mode SeekVLN policy, not AuxThink base."""

    return os.environ.get("SEEKVLN_EXTEND_CONTROL_TOKENS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def seekvln_token_id_config(token_ids: SeekVLNTokenIds) -> dict[str, list[int] | Mapping[str, int]]:
    """Serialize token ids for runtime configs and smoke-test reporting."""

    return {
        "control": dict(token_ids.control),
        "mode": list(token_ids.mode),
        "response_special": list(token_ids.response_special),
    }


def initialize_seekvln_control_embeddings(
    model: Any,
    tokenizer: Any,
    token_ids: SeekVLNTokenIds,
) -> dict[str, dict[str, float | int | str]]:
    """Randomly initialize newly added SeekVLN control-token rows.

    HuggingFace ``resize_token_embeddings`` already creates rows for newly
    appended tokens.  This helper makes the intended contract explicit and only
    touches the five Methodology-3.3 control tokens that were absent from the
    AuxThink base tokenizer.  The VILA sentinel is intentionally left to the
    default resize initialization because it is only a preprocessing marker.
    """

    if not token_ids.added_control_tokens:
        return {}
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise ValueError("SeekVLN control-token initialization requires input embeddings")
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("SeekVLN control-token initialization requires output embeddings")
    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    if input_weight.shape[0] != len(tokenizer) or output_weight.shape[0] != len(tokenizer):
        raise ValueError(
            "Tokenizer and embedding sizes differ before SeekVLN token initialization: "
            f"tokenizer={len(tokenizer)}, input={input_weight.shape[0]}, output={output_weight.shape[0]}"
        )

    llm = model.get_llm() if hasattr(model, "get_llm") else model
    initializer_range = float(getattr(getattr(llm, "config", None), "initializer_range", 0.02))
    audit: dict[str, dict[str, float | int | str]] = {}
    with torch.no_grad():
        for token in token_ids.added_control_tokens:
            token_id = _single_token_id(tokenizer, token)
            input_weight[token_id].normal_(mean=0.0, std=initializer_range)
            output_weight[token_id].normal_(mean=0.0, std=initializer_range)
            audit[token] = {
                "target_token_id": int(token_id),
                "method": "random_normal",
                "mean": 0.0,
                "std": initializer_range,
            }
    return audit
