"""Methodology-3.3 evaluation metric aggregation for SeekVLN rollouts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def aggregate_seekvln_decision_metrics(decisions: list[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate SeekVLN-specific rates and available VLN metrics.

    The input is a flat list of decision records from the agent loop.  Standard
    VLN metrics are read from the final decision of each episode when present.
    """

    if not decisions:
        return {
            "seekvln/episode_count": 0.0,
            "seekvln/decision_count": 0.0,
            "seekvln/nav_rate": 0.0,
            "seekvln/seek_rate": 0.0,
            "seekvln/format_accuracy": 0.0,
            "seekvln/invalid_trigger_rate": 0.0,
        }

    total = float(len(decisions))
    mode_counts = Counter(str(item.get("mode") or item.get("selected_mode") or "") for item in decisions)
    valid_count = sum(1 for item in decisions if bool(item.get("format_valid", False)))
    invalid_reasons = Counter(
        str(item.get("invalid_reason") or "format_invalid")
        for item in decisions
        if not bool(item.get("format_valid", False))
    )

    result = {
        "seekvln/episode_count": float(len(_final_decisions_by_episode(decisions))),
        "seekvln/decision_count": total,
        "seekvln/nav_rate": float(mode_counts.get("nav", 0)) / total,
        "seekvln/seek_rate": float(mode_counts.get("seek", 0)) / total,
        "seekvln/format_accuracy": float(valid_count) / total,
        "seekvln/invalid_trigger_rate": float(total - valid_count) / total,
    }
    for reason, count in sorted(invalid_reasons.items()):
        result[f"seekvln/invalid_reason/{reason}"] = float(count)

    final_decisions = list(_final_decisions_by_episode(decisions).values())
    for output_name, source_names in (
        ("ne", ("distance_to_goal", "navigation_error", "NE")),
        ("sr", ("success",)),
        ("spl", ("spl",)),
        ("ndtw", ("ndtw", "nDTW")),
        ("osr", ("oracle_success", "oracle_success_rate", "oracle_sr")),
    ):
        values = [
            value
            for item in final_decisions
            for value in [_first_metric_value(item, source_names)]
            if value is not None
        ]
        if values:
            result[f"seekvln/{output_name}"] = sum(values) / float(len(values))
    return result


def _final_decisions_by_episode(decisions: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_episode: dict[str, Mapping[str, Any]] = {}
    for fallback_index, item in enumerate(decisions):
        episode_id = str(item.get("episode_id", "__single_episode__"))
        current = by_episode.get(episode_id)
        if current is None or _decision_index(item, fallback_index) >= _decision_index(current, -1):
            by_episode[episode_id] = item
    return by_episode


def _decision_index(item: Mapping[str, Any], fallback: int) -> int:
    for key in ("decision_index", "environment_decision_index"):
        if key in item and item[key] is not None:
            return int(item[key])
    return int(fallback)


def _first_metric_value(item: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    containers = (item, item.get("current") or {}, item.get("metrics") or {})
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for name in names:
            if name in container and container[name] is not None:
                return float(container[name])
    return None
