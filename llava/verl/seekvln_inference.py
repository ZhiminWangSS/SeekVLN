"""Methodology-3.3 inference helpers for SeekVLN VERL rollouts."""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Sequence

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token


A3_PROMPT_SUFFIX = (
    "Analyze this series of images and begin your response with exactly one mode token: "
    "<nav> or <seek>. Use <nav> to predict the next action, which could be turning left "
    "or right by a specific degree, moving forward a certain distance, or stopping if "
    "the task is completed. Use <seek> when additional task-relevant visual evidence is "
    "needed to reliably determine the task progress and next action. The environment "
    "will then provide left, forward, and right views enclosed by <seek></seek>. Based "
    "on these views, reason about the completed tasks, next task, and key evidence "
    "within <think></think>, and then predict the next action with <nav>."
)

SEEK_ENVIRONMENT_TURN = "left view:<image> forward view:<image> right view:<image></seek><think>"


@dataclass(frozen=True)
class SeekVLNParsedResponse:
    mode: str
    text: str
    final_action_text: str
    actions: list[int]
    format_valid: bool
    invalid_reason: str | None = None
    action_fallback: bool = False


def build_seekvln_stage1_prompt(instruction: str, image_count: int) -> str:
    if int(image_count) < 1:
        raise ValueError("SeekVLN stage-1 prompt requires at least one image")
    instruction = " ".join(str(instruction).split())
    if int(image_count) == 1:
        observation_text = "current observation:<image>."
    else:
        history = "<image>" * (int(image_count) - 1)
        observation_text = f"historical observations: {history} and current observation:<image>."
    return (
        "Imagine you are a robot programmed for navigation tasks. You have been "
        f"given a video of {observation_text} Your assigned task is: {instruction}. "
        f"{A3_PROMPT_SUFFIX}"
    )


def tokenize_seekvln_conversation(tokenizer, conversation: Sequence[dict], add_generation_prompt: bool = True):
    messages = []
    for turn in conversation:
        sender = turn["from"]
        if sender == "human":
            role = "user"
        elif sender == "gpt":
            role = "assistant"
        else:
            raise ValueError(f"Unexpected SeekVLN conversation role: {sender}")
        messages.append({"role": role, "content": str(turn["value"]).strip()})
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)


def _auxthink_random_fallback_action() -> list[int]:
    return [random.randint(1, 3)]


def _parse_seekvln_action_with_status(text: str) -> tuple[list[int], bool]:
    """Parse final action text with AuxThink's original parser semantics.

    The comma truncation is intentionally applied only after SeekVLN has already
    isolated the final action span after ``<nav>``.  It must not be applied to
    the earlier ``<think>`` content, which may naturally contain commas.
    """

    output = str(text).split(",", 1)[0]
    if re.search(r"stop", output, re.I):
        return [0], False

    if re.search(r"left", output, re.I):
        action, unit = 2, 15
    elif re.search(r"right", output, re.I):
        action, unit = 3, 15
    elif re.search(r"forward", output, re.I):
        action, unit = 1, 25
    else:
        return _auxthink_random_fallback_action(), True

    match = re.search(r"-?\d+", output)
    if match is None:
        return _auxthink_random_fallback_action(), True
    count = min(3, int(float(match.group()) / unit))
    if count <= 0:
        return _auxthink_random_fallback_action(), True
    return [action] * count, False


def parse_seekvln_action(text: str) -> list[int]:
    actions, _ = _parse_seekvln_action_with_status(text)
    return actions


def parse_seekvln_response(mode: str, text: str) -> SeekVLNParsedResponse:
    mode = str(mode)
    text = str(text).strip()
    if mode == "nav":
        full_text = "<nav>" + text if not text.startswith("<nav>") else text
        action_text = full_text.split("<nav>", 1)[1]
        actions, action_fallback = _parse_seekvln_action_with_status(action_text)
        invalid_reason = "unparseable_action" if action_fallback else None
        return SeekVLNParsedResponse(
            mode=mode,
            text=full_text,
            final_action_text=action_text.strip(),
            actions=actions,
            format_valid=not action_fallback,
            invalid_reason=invalid_reason,
            action_fallback=action_fallback,
        )
    if mode != "seek":
        return SeekVLNParsedResponse(
            mode=mode,
            text=text,
            final_action_text="",
            actions=[],
            format_valid=False,
            invalid_reason="unknown_mode",
            action_fallback=False,
        )
    full_text = "<seek>" + SEEK_ENVIRONMENT_TURN + text
    match = re.search(r"</think><nav>(?P<action>.*)$", full_text, re.S)
    action_text = match.group("action") if match else ""
    actions, action_fallback = _parse_seekvln_action_with_status(action_text) if match else ([], False)
    invalid_reason = None
    if "</seek><think>" not in full_text:
        invalid_reason = "missing_seek_close_think_open"
    elif not match:
        invalid_reason = "missing_think_close_nav"
    elif "Completed tasks:" not in full_text:
        invalid_reason = "missing_completed_tasks"
    elif "Next task:" not in full_text:
        invalid_reason = "missing_next_task"
    elif "Key Evidence:" not in full_text:
        invalid_reason = "missing_key_evidence"
    elif action_fallback:
        invalid_reason = "unparseable_action"
    format_valid = invalid_reason is None
    return SeekVLNParsedResponse(
        mode=mode,
        text=full_text,
        final_action_text=action_text.strip(),
        actions=actions,
        format_valid=format_valid,
        invalid_reason=invalid_reason,
        action_fallback=action_fallback,
    )
